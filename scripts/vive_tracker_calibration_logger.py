"""Print full Vive tracker poses for mounting calibration; never open motor serial."""

import json
import math
import time

import openvr  # Included in the project dependencies.


def normalize_vector(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Return (x, y, z) normalized to unit length."""
    magnitude = math.sqrt(x * x + y * y + z * z)

    if magnitude == 0:
        return 0.0, 0.0, 0.0

    return x / magnitude, y / magnitude, z / magnitude


def main() -> None:
    try:
        # 1. Initialize OpenVR as a background application
        vr_system = openvr.init(openvr.VRApplication_Background)
    except openvr.OpenVRError as e:
        print(f"Failed to initialize OpenVR: {e}")
        print("Tip: Make sure SteamVR is running.")
        return

    print("Successfully connected to SteamVR.")
    print("Scanning for Vive Trackers... Press Ctrl+C to exit.\n")

    try:
        while True:
            # 2. Get poses for all connected devices
            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding,
                0.0,
                openvr.k_unMaxTrackedDeviceCount
            )

            # 3. Loop through all possible device slots
            for device_id in range(openvr.k_unMaxTrackedDeviceCount):
                pose = poses[device_id]

                if not pose.bDeviceIsConnected:
                    continue

                # Check if the device is specifically a Vive Tracker
                device_class = vr_system.getTrackedDeviceClass(device_id)

                if device_class == openvr.TrackedDeviceClass_GenericTracker:

                    # Ensure tracking data is currently valid
                    if pose.bPoseIsValid and pose.eTrackingResult == openvr.TrackingResult_Running_OK:
                        matrix = pose.mDeviceToAbsoluteTracking

                        # -----------------------------------------
                        # Position
                        # -----------------------------------------
                        x = matrix[0][3]
                        y = matrix[1][3]
                        z = matrix[2][3]

                        # -----------------------------------------
                        # Orientation
                        #
                        # The upper-left 3x3 portion is the
                        # rotation matrix.
                        #
                        # OpenVR convention:
                        #   +X = right
                        #   +Y = up
                        #   -Z = forward
                        #
                        # Therefore, the device's forward vector
                        # in world coordinates is the negative
                        # of the third column.
                        # -----------------------------------------
                        forward_x = -matrix[0][2]
                        forward_y = -matrix[1][2]
                        forward_z = -matrix[2][2]

                        # Normalize it to a unit vector
                        orientation_x, orientation_y, orientation_z = (
                            normalize_vector(
                                forward_x,
                                forward_y,
                                forward_z
                            )
                        )

                        # -----------------------------------------
                        # Serial number
                        # -----------------------------------------
                        try:
                            serial_number = (
                                vr_system.getStringTrackedDeviceProperty(
                                    device_id,
                                    openvr.Prop_SerialNumber_String
                                )
                            )
                        except Exception:
                            serial_number = "UNKNOWN_SERIAL"

                        # -----------------------------------------
                        # Controller/device type
                        # -----------------------------------------
                        try:
                            role = (
                                vr_system.getStringTrackedDeviceProperty(
                                    device_id,
                                    openvr.Prop_ControllerType_String
                                )
                            )
                        except Exception:
                            role = "Unknown"

                        print(
                            f"Tracker [{serial_number}] ({role}) -> "
                            f"Position: "
                            f"X: {x:.3f}, "
                            f"Y: {y:.3f}, "
                            f"Z: {z:.3f} | "
                            f"Forward direction: "
                            f"X: {orientation_x:.3f}, "
                            f"Y: {orientation_y:.3f}, "
                            f"Z: {orientation_z:.3f}"
                        )

                        # Keep the complete device-to-world rotation for calibration.
                        # Full precision; no Euler-angle or axis-order ambiguity.
                        print("Full pose JSON: " + json.dumps({
                            "serial": serial_number,
                            "position": [float(x), float(y), float(z)],
                            "rotation_matrix": [
                                [float(matrix[row][column]) for column in range(3)]
                                for row in range(3)
                            ]
                        }, allow_nan=False), flush=True)

            # Polling frequency (10 Hz)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting Tracker Logger.")

    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()
