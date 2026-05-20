#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "robot.hpp"
#include "motor.hpp"
#include "serial_driver.hpp"
#include "parse_robot_params.hpp"
#include "serial_struct.hpp"

namespace py = pybind11;
using namespace hightorque_robot;

PYBIND11_MODULE(_hightorque_robot, m) {
    m.doc() = "Python interface for the HighTorque robot motor controller";

    // ==================== Enums ====================

    // Motor type enum
    py::enum_<motor_type>(m, "MotorType")
        .value("M3536_32", motor_type::m3536_32)
        .value("M4538_19", motor_type::m4538_19)
        .value("M5046_20", motor_type::m5046_20)
        .value("M5047_09", motor_type::m5047_09)
        .value("M5047_36", motor_type::m5047_36)
        .value("M5047_36_2", motor_type::m5047_36_2)
        .value("M4438_30", motor_type::m4438_30)
        .value("M4438_32", motor_type::m4438_32)
        .value("M6056_36", motor_type::m6056_36)
        .value("M5043_20", motor_type::m5043_20)
        .value("M7256_35", motor_type::m7256_35)
        .value("M60SG_35", motor_type::m60sg_35)
        .value("M60BM_35", motor_type::m60bm_35)
        .value("MGENERAL", motor_type::mGeneral)
        .export_values();

    // Position/velocity conversion type
    py::enum_<pos_vel_convert_type>(m, "PosVelConvertType")
        .value("RADIAN_2PI", pos_vel_convert_type::radian_2pi, "Radians (0-2 pi)")
        .value("ANGLE_360", pos_vel_convert_type::angle_360, "Degrees (0-360)")
        .value("TURNS", pos_vel_convert_type::turns, "Turns")
        .export_values();

    // ==================== Data structures ====================

    // Motor state struct
    py::class_<motor_back_t>(m, "MotorState")
        .def(py::init<>())
        .def_readwrite("time", &motor_back_t::time, "Timestamp")
        .def_readwrite("ID", &motor_back_t::ID, "Motor ID")
        .def_readwrite("mode", &motor_back_t::mode, "Operating mode")
        .def_readwrite("fault", &motor_back_t::fault, "Fault code")
        .def_readwrite("position", &motor_back_t::position, "Position (rad)")
        .def_readwrite("velocity", &motor_back_t::velocity, "Velocity (rad/s)")
        .def_readwrite("torque", &motor_back_t::torque, "Torque (N.m)")
        .def("__repr__", [](const motor_back_t &s) {
            return "<MotorState ID=" + std::to_string(s.ID) +
                   " pos=" + std::to_string(s.position) +
                   " vel=" + std::to_string(s.velocity) +
                   " torque=" + std::to_string(s.torque) + ">";
        });

    // Motor version info
    py::class_<cdc_rx_motor_version_s>(m, "MotorVersion")
        .def(py::init<>())
        .def_readwrite("id", &cdc_rx_motor_version_s::id, "Motor ID")
        .def_readwrite("major", &cdc_rx_motor_version_s::major, "Major version")
        .def_readwrite("minor", &cdc_rx_motor_version_s::minor, "Minor version")
        .def_readwrite("patch", &cdc_rx_motor_version_s::patch, "Patch version")
        .def("__repr__", [](const cdc_rx_motor_version_s &v) {
            return "<MotorVersion " + std::to_string(v.id) + " v" +
                   std::to_string(v.major) + "." +
                   std::to_string(v.minor) + "." +
                   std::to_string(v.patch) + ">";
        });

    // Configuration parameter structs
    py::class_<MotorParams>(m, "MotorParams")
        .def(py::init<>())
        .def_readwrite("type", &MotorParams::type)
        .def_readwrite("id", &MotorParams::id)
        .def_readwrite("name", &MotorParams::name)
        .def_readwrite("num", &MotorParams::num)
        .def_readwrite("pos_limit_enable", &MotorParams::pos_limit_enable)
        .def_readwrite("pos_upper", &MotorParams::pos_upper)
        .def_readwrite("pos_lower", &MotorParams::pos_lower)
        .def_readwrite("tor_limit_enable", &MotorParams::tor_limit_enable)
        .def_readwrite("tor_upper", &MotorParams::tor_upper)
        .def_readwrite("tor_lower", &MotorParams::tor_lower);

    py::class_<CANPortParams>(m, "CANPortParams")
        .def(py::init<>())
        .def_readwrite("serial_id", &CANPortParams::serial_id)
        .def_readwrite("motor_num", &CANPortParams::motor_num)
        .def_readwrite("motors", &CANPortParams::motors);

    py::class_<CANBoardParams>(m, "CANBoardParams")
        .def(py::init<>())
        .def_readwrite("CANport_num", &CANBoardParams::CANport_num)
        .def_readwrite("CANports", &CANBoardParams::CANports);

    py::class_<RobotParams>(m, "RobotParams")
        .def(py::init<>())
        .def_readwrite("motor_timeout_ms", &RobotParams::motor_timeout_ms)
        .def_readwrite("robot_name", &RobotParams::robot_name)
        .def_readwrite("Serial_Type", &RobotParams::Serial_Type)
        .def_readwrite("Seial_baudrate", &RobotParams::Seial_baudrate)
        .def_readwrite("CANboard_num", &RobotParams::CANboard_num)
        .def_readwrite("CANboards", &RobotParams::CANboards);

    // ==================== Motor class ====================

    py::class_<motor>(m, "Motor")
        // Note: the motor class constructor takes many arguments and is not
        // intended to be created directly from Python. Get motor references
        // from a robot object instead.

        // Basic control methods
        .def("position", &motor::position, py::arg("position"),
             "Position control\n\nArgs:\n  position: Target position (rad)")
        .def("velocity", &motor::velocity, py::arg("velocity"),
             "Velocity control\n\nArgs:\n  velocity: Target velocity (rad/s)")
        .def("torque", &motor::torque, py::arg("torque"),
             "Torque control\n\nArgs:\n  torque: Target torque (N.m)")
        .def("voltage", &motor::voltage, py::arg("voltage"),
             "Voltage control\n\nArgs:\n  voltage: Target voltage (V)")
        .def("current", &motor::current, py::arg("current"),
             "Current control\n\nArgs:\n  current: Target current (A)")

        // Hybrid control methods
        .def("pos_vel_MAXtqe", &motor::pos_vel_MAXtqe,
             py::arg("position"), py::arg("velocity"), py::arg("torque_max"),
             "Position + velocity + max-torque control\n\n"
             "Args:\n"
             "  position: Target position (rad)\n"
             "  velocity: Target velocity (rad/s)\n"
             "  torque_max: Max-torque limit (N.m)")
        .def("pos_vel_tqe_kp_kd", &motor::pos_vel_tqe_kp_kd,
             py::arg("position"), py::arg("velocity"), py::arg("torque"),
             py::arg("kp"), py::arg("kd"),
             "Five-parameter control: position + velocity + torque + Kp + Kd\n\n"
             "Args:\n"
             "  position: Target position (rad)\n"
             "  velocity: Target velocity (rad/s)\n"
             "  torque: Feed-forward torque (N.m)\n"
             "  kp: PID proportional gain\n"
             "  kd: PID derivative gain")
        .def("pos_vel_kp_kd", &motor::pos_vel_kp_kd,
             py::arg("position"), py::arg("velocity"), py::arg("kp"), py::arg("kd"),
             "Position + velocity + PID gains control")
        .def("pos_vel_acc", &motor::pos_vel_acc,
             py::arg("position"), py::arg("velocity"), py::arg("acc"),
             "Position + velocity + acceleration control")

        // Operations
        .def("stop", &motor::stop, "Stop the motor")
        .def("brake", &motor::brake, "Brake the motor")
        .def("reset", &motor::reset, "Reset the motor")
        .def("send_state_cmd", &motor::send_state_cmd, "Send a state query command")

        // Query methods
        .def("get_motor_id", &motor::get_motor_id, "Get the motor ID")
        .def("get_motor_enum_type", &motor::get_motor_enum_type, "Get the motor type")
        .def("get_motor_num", &motor::get_motor_num, "Get the motor number")
        .def("get_motor_name", &motor::get_motor_name, "Get the motor name")
        .def("get_current_motor_state", &motor::get_current_motor_state,
             py::return_value_policy::reference,
             "Get the current motor state\n\nReturns: MotorState object")
        .def("get_version", &motor::get_version,
             py::return_value_policy::reference,
             "Get the motor version info\n\nReturns: MotorVersion object")

        // Limit flags (read-only)
        .def_readonly("pos_limit_flag", &motor::pos_limit_flag,
                     "Position-limit flag: 0=normal, 1=above upper, -1=below lower")
        .def_readonly("tor_limit_flag", &motor::tor_limit_flag,
                     "Torque-limit flag: 0=normal, 1=above upper")

        .def("__repr__", [](motor &m) {
            return "<Motor ID=" + std::to_string(m.get_motor_id()) +
                   " name='" + m.get_motor_name() + "'>";
        });

    // ==================== Robot class ====================

    py::class_<robot>(m, "Robot")
        .def(py::init<>(),
             "Create a robot instance (using the default configuration)")
        .def(py::init<const std::string&>(),
             py::arg("config_path"),
             "Create a robot instance\n\n"
             "Args:\n"
             "  config_path: Path to the YAML configuration file")

        // Motor control
        .def("motor_send_cmd", &robot::motor_send_cmd,
             "Send the motor control commands to all motors.\n"
             "Must be called after setting the motor control parameters.")
        .def("set_stop", &robot::set_stop,
             "Stop all motors")
        .def("set_reset", &robot::set_reset,
             "Reset all motors")
        .def("send_get_motor_state_cmd", &robot::send_get_motor_state_cmd,
             "Send a state query command to all motors")
        .def("send_get_motor_version_cmd", &robot::send_get_motor_version_cmd,
             "Send a state query command to all motors")
        .def("set_reset_zero",
             static_cast<void (robot::*)()>(&robot::set_reset_zero),
             "Reset the zero point of all motors")
        .def("set_reset_zero_motors",
             static_cast<void (robot::*)(std::initializer_list<int>)>(&robot::set_reset_zero),
             py::arg("motor_ids"),
             "Reset the zero point of the specified motors\n\nArgs:\n  motor_ids: List of motor IDs")
        .def("set_timeout",
             static_cast<void (robot::*)(int16_t)>(&robot::set_timeout),
             py::arg("timeout_ms"),
             "Set the motor timeout\n\nArgs:\n  timeout_ms: Timeout (milliseconds)")

        // LCM-related
        .def("lcm_enable", &robot::lcm_enable,
             "Enable LCM message publishing.\n"
             "Starts a background thread that publishes the robot state to LCM.")
        .def("publishJointStates", &robot::publishJointStates,
             "Manually publish a single joint-state message to LCM")

        // Advanced functionality
        .def("motor_version_detection", &robot::motor_version_detection,
             "Detect the version of all motors")
        .def("canboard_fdcan_reset", &robot::canboard_fdcan_reset,
             "Re-initialize FDCAN communication")

        // Accessors
        .def("get_motors", [](robot &r) {
            return r.Motors;
        }, py::return_value_policy::reference,
           "Get the list of all motor objects\n\nReturns: list of Motor objects")

        .def("get_motor_by_id", [](robot &r, int motor_id) -> motor* {
            for (auto* m : r.Motors) {
                if (m->get_motor_id() == motor_id) {
                    return m;
                }
            }
            throw std::runtime_error("No motor found with ID " + std::to_string(motor_id));
        }, py::arg("motor_id"),
           py::return_value_policy::reference,
           "Get a motor object by ID\n\nArgs:\n  motor_id: Motor ID\n\nReturns: Motor object")

        .def("get_motor_by_name", [](robot &r, const std::string& name) -> motor* {
            for (auto* m : r.Motors) {
                if (m->get_motor_name() == name) {
                    return m;
                }
            }
            throw std::runtime_error("No motor found with name '" + name + "'");
        }, py::arg("name"),
           py::return_value_policy::reference,
           "Get a motor object by name\n\nArgs:\n  name: Motor name\n\nReturns: Motor object")

        .def_readonly("robot_params", &robot::robot_params,
                     "Robot configuration parameters (RobotParams object)")
        .def_readonly("motor_timeout_ms", &robot::motor_timeout_ms,
                     "Motor timeout (milliseconds)")

        .def("__repr__", [](robot &r) {
            return "<Robot '" + r.robot_params.robot_name +
                   "' with " + std::to_string(r.Motors.size()) + " motors>";
        });

    // ==================== Helper functions ====================

    m.def("parse_robot_params", &parseRobotParams,
          py::arg("file_path"),
          "Parse robot parameters from a YAML file\n\n"
          "Args:\n"
          "  file_path: Path to the YAML configuration file\n\n"
          "Returns: RobotParams object");

    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__cpp_sdk_version__") = "4.4.7";
}
