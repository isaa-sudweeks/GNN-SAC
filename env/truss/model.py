import mujoco
import numpy as np
import xml.etree.ElementTree as ET


def _load_mjx():
    try:
        import jax
        import jax.numpy as jnp
        from mujoco import mjx
    except ImportError as exc:
        raise ImportError(
            "The MJX backend requires JAX and mujoco.mjx. Install the project "
            "dependencies with JAX support before using mujoco_backend='mjx'."
        ) from exc
    return jax, jnp, mjx


class MujocoModel:
    def __init__(self, xml_path, backend="mjx"):
        self.xml_path = xml_path
        self.backend = backend
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._jax = None
        self._jnp = None
        self._mjx = None
        self._mjx_model = None
        self._mjx_data = None
        self._compiled_steps = {}
        self._load_model_metadata(xml_path)
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        self.ctrl_home = np.zeros(self.model.nu)
        self.act_home = np.ones(self.model.na)
        mujoco.mj_forward(self.model, self.data)
        self._init_backend()
        self.initial_critical_eig = max(self._critical_eig(), 1e-8)

    def _init_backend(self):
        if self.backend in ("mujoco", "native", None):
            self.backend = "mujoco"
            return
        if self.backend != "mjx":
            raise ValueError(f"Unsupported MuJoCo backend '{self.backend}'. Use 'mjx' or 'mujoco'.")

        self._jax, self._jnp, self._mjx = _load_mjx()
        self._mjx_model = self._mjx.put_model(self.model)
        self._mjx_data = self._mjx.put_data(self.model, self.data)

    @property
    def uses_mjx(self):
        return self.backend == "mjx"

    def _to_numpy(self, value):
        return np.asarray(value)

    def _data_array(self, name):
        if self.uses_mjx:
            return self._to_numpy(getattr(self._mjx_data, name))
        return getattr(self.data, name)

    def _replace_mjx_data(self, **kwargs):
        self._mjx_data = self._mjx_data.replace(**kwargs)

    def _sync_mjx_to_mujoco_data(self):
        if not self.uses_mjx:
            return
        try:
            self.data = self._mjx.get_data(self.model, self._mjx_data)
        except AttributeError:
            self.data.qpos[:] = self._data_array("qpos")
            self.data.qvel[:] = self._data_array("qvel")
            self.data.ctrl[:] = self._data_array("ctrl")
            if self.model.na:
                self.data.act[:] = self._data_array("act")
            mujoco.mj_forward(self.model, self.data)

    def sync_for_render(self):
        self._sync_mjx_to_mujoco_data()

    def get_ctrl(self):
        return self._data_array("ctrl").copy()

    def set_ctrl(self, ctrl):
        ctrl = np.asarray(ctrl, dtype=np.float32)
        if ctrl.shape != (self.model.nu,):
            ctrl = np.reshape(ctrl, (self.model.nu,))
        if self.uses_mjx:
            self._replace_mjx_data(ctrl=self._jnp.asarray(ctrl))
            self.data.ctrl[:] = ctrl
        else:
            self.data.ctrl[:] = ctrl

    def _compile_stepper(self, nsubsteps):
        if nsubsteps in self._compiled_steps:
            return self._compiled_steps[nsubsteps]

        mjx_model = self._mjx_model
        mjx = self._mjx

        @self._jax.jit
        def step_n(data, ctrl):
            data = data.replace(ctrl=ctrl)

            def body(_, current_data):
                return mjx.step(mjx_model, current_data)

            return self._jax.lax.fori_loop(0, nsubsteps, body, data)

        self._compiled_steps[nsubsteps] = step_n
        return step_n

    def advance(self, nsubsteps=1, viewer=None):
        if self.uses_mjx:
            ctrl = self._jnp.asarray(self._data_array("ctrl"))
            step_n = self._compile_stepper(int(nsubsteps))
            self._mjx_data = step_n(self._mjx_data, ctrl)
            self._mjx_data = self._jax.block_until_ready(self._mjx_data)
            if viewer is not None:
                self._sync_mjx_to_mujoco_data()
                viewer.sync()
            return

        for _ in range(int(nsubsteps)):
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

    def _load_model_metadata(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        self.node_names = []
        self.node_axes = {}
        self.node_body_ids = {}
        self.site_to_node = {}

        def dominant_axis(axis_str):
            axis = np.fromstring(axis_str, sep=" ", dtype=float)
            if axis.size != 3:
                raise ValueError(f"Invalid joint axis '{axis_str}' in {xml_path}")
            return "xyz"[int(np.argmax(np.abs(axis)))]

        def visit_body(body_elem, inherited_node=None):
            body_name = body_elem.get("name")
            current_node = inherited_node

            if body_name and body_name.startswith("node_"):
                current_node = body_name
                self.node_names.append(body_name)
                joint_axes = []
                for joint in body_elem.findall("joint"):
                    if joint.get("type", "hinge") == "slide":
                        joint_axes.append(dominant_axis(joint.get("axis", "0 0 0")))
                self.node_axes[body_name] = tuple(sorted(joint_axes, key="xyz".index))
                self.node_body_ids[body_name] = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_name,
                )

            if current_node is not None:
                for site in body_elem.findall("site"):
                    site_name = site.get("name")
                    if site_name:
                        self.site_to_node[site_name] = current_node

            for child_body in body_elem.findall("body"):
                visit_body(child_body, current_node)

        worldbody = root.find("worldbody")
        if worldbody is not None:
            for body in worldbody.findall("body"):
                visit_body(body)

        self.node_names = sorted(self.node_names, key=lambda name: int(name.split("_")[1]))
        self.active_axes = self.node_axes[self.node_names[0]] if self.node_names else ("x", "z")
        self.axis_indices = tuple("xyz".index(axis) for axis in self.active_axes)

        structural_tendon_names = set()
        actuator = root.find("actuator")
        if actuator is not None:
            for actuator_elem in actuator:
                tendon_name = actuator_elem.get("tendon")
                if tendon_name:
                    structural_tendon_names.add(tendon_name)

        equality = root.find("equality")
        if equality is not None:
            for constraint in equality.findall("tendon"):
                tendon_name = constraint.get("tendon1")
                if tendon_name:
                    structural_tendon_names.add(tendon_name)

        tendon_defs = {}
        tendon_root = root.find("tendon")
        if tendon_root is not None:
            for spatial in tendon_root.findall("spatial"):
                sites = [site_ref.get("site") for site_ref in spatial.findall("site")]
                tendon_defs[spatial.get("name")] = [site for site in sites if site]

        self.structural_edges = []
        for tendon_name in sorted(structural_tendon_names):
            sites = tendon_defs.get(tendon_name, [])
            if len(sites) != 2:
                continue
            node_pair = tuple(self.site_to_node.get(site_name) for site_name in sites)
            if None not in node_pair and node_pair[0] != node_pair[1]:
                self.structural_edges.append(node_pair)

    
    def reset(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.data.qpos[:] = self.init_qpos + rng.uniform(-0.005, 0.005, size=self.model.nq)
        self.data.qvel[:] = self.init_qvel + rng.uniform(-0.005, 0.005, size=self.model.nv)
        self.data.ctrl[:] = self.ctrl_home.copy()
        if mujoco.mjtDyn.mjDYN_INTEGRATOR in self.model.actuator_dyntype:
            self.data.act[:] = self.act_home.copy()
        mujoco.mj_forward(self.model, self.data)
        if self.uses_mjx:
            self._mjx_data = self._mjx.put_data(self.model, self.data)
    
    def get_node_loc_dict(self):
        node_dict = {}
        xpos = self._data_array("xpos")
        for i in range(self.model.nbody):
            node_dict[self.model.body(i).name] = xpos[i]
        return node_dict

    def get_node_velocity_dict(self):
        vel_dict = {}
        cvel = self._data_array("cvel")
        for i in range(self.model.nbody):
            vel_dict[self.model.body(i).name] = cvel[i]
        return vel_dict

    def get_edge_length_dict(self):
        tendon_dict = {}
        ten_length = self._data_array("ten_length")
        for ten in range(self.model.ntendon):
            tendon_dict[self.model.tendon(ten).name] = ten_length[ten]
        return tendon_dict

    def get_edge_velocity_dict(self):
        tendon_dict = {}
        ten_velocity = self._data_array("ten_velocity")
        for ten in range(self.model.ntendon):
            tendon_dict[self.model.tendon(ten).name] = ten_velocity[ten]
        return tendon_dict

    def get_node_position_dict(self):
        xpos = self._data_array("xpos")
        return {
            node_name: xpos[self.node_body_ids[node_name]].copy()
            for node_name in self.node_names
        }

    def get_node_velocity_linear_dict(self):
        cvel = self._data_array("cvel")
        return {
            node_name: cvel[self.node_body_ids[node_name]][3:].copy()
            for node_name in self.node_names
        }

    def get_node_position_matrix(self):
        xpos = self._data_array("xpos")
        return np.array([xpos[self.node_body_ids[node_name]] for node_name in self.node_names])

    def get_node_linear_velocity_matrix(self):
        cvel = self._data_array("cvel")
        return np.array([cvel[self.node_body_ids[node_name]][3:] for node_name in self.node_names])

    def _rigidity_matrix(self):
        dims = len(self.active_axes)
        num_nodes = len(self.node_names)
        node_positions = self.get_node_position_dict()
        rows = []

        for node_a, node_b in self.structural_edges:
            pa = node_positions[node_a][list(self.axis_indices)]
            pb = node_positions[node_b][list(self.axis_indices)]
            delta = pb - pa
            length = np.linalg.norm(delta)
            if length < 1e-8:
                continue

            direction = delta / length
            row = np.zeros(num_nodes * dims, dtype=float)
            ia = self.node_names.index(node_a) * dims
            ib = self.node_names.index(node_b) * dims
            row[ia:ia + dims] = -direction
            row[ib:ib + dims] = direction
            rows.append(row)

        if not rows:
            return np.zeros((0, num_nodes * dims), dtype=float)
        return np.vstack(rows)

    def _critical_eig(self):
        rigidity_matrix = self._rigidity_matrix()
        if rigidity_matrix.size == 0:
            return 0.0

        eigvals = np.linalg.eigvalsh(rigidity_matrix.T @ rigidity_matrix)
        eigvals = np.sort(np.real(eigvals))
        dims = len(self.active_axes)
        rigid_body_modes = dims + (dims * (dims - 1)) // 2
        if eigvals.size <= rigid_body_modes:
            return 0.0
        return float(max(eigvals[rigid_body_modes], 0.0))

    def collapse_check(self):
        return self._critical_eig() / self.initial_critical_eig

    def get_forward_velocity_x(self):
        linear_velocities = self.get_node_linear_velocity_matrix()
        return float(np.mean(linear_velocities[:, 0]))

    def get_forward_velocity_y(self):
        linear_velocities = self.get_node_linear_velocity_matrix()
        return float(np.mean(linear_velocities[:, 1]))

    def get_slip_penalty(self, height=0.2, axis="x"):
        positions = self.get_node_position_matrix()
        linear_velocities = self.get_node_linear_velocity_matrix()
        contact_mask = positions[:, 2] < height
        axis_idx = "xyz".index(axis)
        return float(np.sum(np.abs(linear_velocities[contact_mask, axis_idx])))
