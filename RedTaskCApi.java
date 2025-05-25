package opmode.auto;

import android.annotation.SuppressLint;

import com.arcrobotics.ftclib.hardware.motors.Motor;
import com.arcrobotics.ftclib.hardware.motors.MotorEx;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.hardware.DcMotorEx;
import com.qualcomm.robotcore.hardware.DcMotorSimple;
import com.qualcomm.robotcore.hardware.HardwareMap;
import com.qualcomm.robotcore.hardware.Servo;
import com.qualcomm.robotcore.util.ElapsedTime;
import com.qualcomm.robotcore.util.Range;

import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.*;

import java.util.*;
import java.lang.Math;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class RedTaskCApi {

    // --- Vision Constants ---
    private static final String WEBCAM_NAME = "Webcam";
    private static final OpenCvCameraRotation CAMERA_STREAMING_ROTATION = OpenCvCameraRotation.UPSIDE_DOWN;
    private static final boolean IMAGE_ROTATE_180 = true;
    private static final double DOWNSCALE_FACTOR = 1.0;
    private static final double PIXELS_TO_CM_RATIO = 0.033984375;
    private static final double CIRCLE_CENTER_OFFSET_CM_X = 0.0;
    private static final double CIRCLE_CENTER_OFFSET_CM_Y = 5.5;
    private static final double ARM_BACK_CM = 11.9;
    private static final double ERROR_CM = 5.0;
    private static final double GRASPABLE_ANGLE_DEG = 120.0;
    private static final List<Scalar[]> RED_HSV_RANGES = List.of(
            new Scalar[]{new Scalar(170, 80, 110), new Scalar(180, 255, 255)},
            new Scalar[]{new Scalar(0, 80, 110), new Scalar(10, 255, 255)}
    );
    private static final double MIN_SIZE_PIXELS = 20000;
    private static final double MAX_SIZE_PIXELS = 1000000;
    private static final int CAMERA_WIDTH = 1280;
    private static final int CAMERA_HEIGHT = 720;

    // --- Servo Constants ---
    private static final double IN_FLIP_INIT_POS = 0.68;
    private static final double IN_FLIP_TURN_INIT_POS = 0.37;
    private static final double CLAW_ROTATE_INIT_POS = 0.63;
    private static final double IN_CLAW_INIT_POS = 0.34;

    private static final double INPUT_TURN_ANGLE_MIN = -GRASPABLE_ANGLE_DEG / 2.0;
    private static final double INPUT_TURN_ANGLE_RANGE_WIDTH = GRASPABLE_ANGLE_DEG;
    private static final double SERVO_TURN_MIN_POS = 0.01;
    private static final double SERVO_TURN_MAX_POS = 0.71;
    private static final double SERVO_TURN_RANGE_WIDTH = SERVO_TURN_MAX_POS - SERVO_TURN_MIN_POS;

    private static final double SERVO_ROTATE_MIN_POS = 0.29;
    private static final double SERVO_ROTATE_MAX_POS = 0.97;

    // --- Slide Motor Constants (MODIFIED FOR 3-STAGE) ---
    private static final String SLIDE_MOTOR_NAME = "hSlideL";
    // Stage definitions
    private static final int SLIDE_STAGE1_TARGET_POSITION = 5000; // End position for stage 1
    private static final double SLIDE_MOVE_SPEED_INITIAL = 0.3;   // Speed for stage 1 (0 -> 7000)

    private static final int SLIDE_INTERMEDIATE_POSITION = 19000; // End position for stage 2
    private static final double SLIDE_MOVE_SPEED_STAGE_2 = 0.3;   // Speed for stage 2 (7000 -> 19000)

    private static final int SLIDE_FINAL_TARGET_POSITION = 24945; // End position for stage 3 (final target)
    private static final double SLIDE_MOVE_SPEED_STAGE_3 = 0.36;  // Speed for stage 3 (19000 -> 24945)

    private static final double SLIDE_EXTEND_TIMEOUT_SECONDS = 1.3; // Timeout for the entire extension

    // --- Drawing Constants ---
    private static final Scalar CIRCLE_COLOR = new Scalar(255, 255, 255);
    private static final int CIRCLE_THICKNESS = 2;
    private static final Scalar PERSISTENT_LINE_COLOR = new Scalar(255, 255, 255);
    private static final int PERSISTENT_LINE_THICKNESS = 2;
    private static final Scalar BOX_COLOR_SELECTED = new Scalar(0, 0, 255);
    private static final Scalar BOX_COLOR_DEFAULT = new Scalar(0, 255, 0);
    private static final Scalar CENTER_TO_TARGET_LINE_COLOR = new Scalar(255, 128, 0);
    private static final int CONNECTION_LINE_THICKNESS = 2;
    private static final Scalar BLOCK_CENTER_CIRCLE_COLOR = new Scalar(255, 255, 0);
    private static final Scalar TEXT_COLOR = new Scalar(255, 255, 0);
    private static final Scalar GRASPABLE_ANGLE_LINE_COLOR = new Scalar(0, 255, 255);
    private static final int GRASPABLE_ANGLE_LINE_THICKNESS = 2;

    // --- Class Members ---
    private HardwareMap hardwareMap = null;
    private Telemetry telemetry = null;
    private OpenCvWebcam webcam = null;
    private BLUEDetectionPipeline BLUEDetectionPipeline = null;
    private Servo inFlip = null;
    private Servo inFlipTurn = null;
    private Servo clawRotate = null;
    private Servo inClaw = null;
    private DcMotorEx hSlideL = null;

    private volatile boolean isInitialized = false;
    private volatile boolean isVisionStreaming = false;
    private volatile boolean isActionSequenceActive = false;
    private volatile ActionState currentActionState = ActionState.IDLE;
    private ElapsedTime stateTimer = new ElapsedTime();
    private ElapsedTime slideMoveTimer = new ElapsedTime();

    private volatile double targetInFlipTurnServoPos = IN_FLIP_TURN_INIT_POS;
    private volatile double targetClawRotateServoPos = CLAW_ROTATE_INIT_POS;

    private enum ActionState {
        IDLE,
        MOVING_SLIDE,
        UPDATING_TURN_ROTATE,
        SETTING_FLIP_UP,
        CLOSING_CLAW,
        SETTING_FLIP_MID
    }

    public RedTaskCApi() {
    }

    @SuppressLint("DefaultLocale")
    public boolean initialize(HardwareMap hwMap, Telemetry tel) {
        if (isInitialized) {
            if (tel != null) tel.addLine("RedTaskCApi: 已初始化。");
            return true;
        }

        this.hardwareMap = hwMap;
        this.telemetry = tel;

        printConfig(); // Print config after setting members

        try {
            inFlip = hardwareMap.get(Servo.class, "inFlip");
            inFlipTurn = hardwareMap.get(Servo.class, "inFlipTurn");
            clawRotate = hardwareMap.get(Servo.class, "clawRotate");
            inClaw = hardwareMap.get(Servo.class, "inClaw");
            logInfo("伺服初始化成功。");
        } catch (Exception e) {
            logError("伺服初始化失败: " + e.getMessage());
            inFlip = null; inFlipTurn = null; clawRotate = null; inClaw = null;
            // Decide if servos are critical for initialization success
        }

        try {
            hSlideL = hardwareMap.get(DcMotorEx.class, SLIDE_MOTOR_NAME);
            hSlideL.setDirection(DcMotorSimple.Direction.REVERSE);
            hSlideL.setMode(DcMotor.RunMode.STOP_AND_RESET_ENCODER);
            // Set mode before setting power or target position
            hSlideL.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
            hSlideL.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
            hSlideL.setPower(0.0); // Ensure motor is stopped initially
            logInfo("滑轨电机 '" + SLIDE_MOTOR_NAME + "' 初始化成功 (RUN_WITHOUT_ENCODER, REVERSED, BRAKE)。");
        } catch (Exception e) {
            logError("滑轨电机 '" + SLIDE_MOTOR_NAME + "' 初始化失败: " + e.getMessage());
            hSlideL = null;
            return false; // Slide motor is likely critical
        }

        int cameraMonitorViewId = hardwareMap.appContext.getResources().getIdentifier(
                "cameraMonitorViewId", "id", hardwareMap.appContext.getPackageName());
        try {
            webcam = OpenCvCameraFactory.getInstance().createWebcam(
                    hardwareMap.get(WebcamName.class, WEBCAM_NAME), cameraMonitorViewId);
        } catch (IllegalArgumentException e) {
            logError("*** 致命错误: 摄像头 '" + WEBCAM_NAME + "' 未找到! ***");
            webcam = null;
            cleanup(); // Clean up already initialized components
            return false; // Vision is critical
        }

        BLUEDetectionPipeline = new BLUEDetectionPipeline();
        webcam.setPipeline(BLUEDetectionPipeline);
        webcam.setMillisecondsPermissionTimeout(5000);

        isInitialized = true;
        logInfo("RedTaskCApi 初始化完成。");
        return true;
    }

    public void startVisionStream() {
        if (!isInitialized || webcam == null || isVisionStreaming) {
            logError("无法启动视觉流。初始化: " + isInitialized + ", 摄像头为 null: " + (webcam == null) + ", 正在流式传输: " + isVisionStreaming);
            return;
        }

        logInfo("正在异步打开摄像头设备...");
        webcam.openCameraDeviceAsync(new OpenCvCamera.AsyncCameraOpenListener() {
            @Override
            public void onOpened() {
                logInfo("摄像头已打开。正在启动 MJPEG 流...");
                try {
                    webcam.startStreaming(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_STREAMING_ROTATION, OpenCvWebcam.StreamFormat.MJPEG);
                    isVisionStreaming = true;
                    logInfo("视觉流成功启动。");
                } catch (Exception e) {
                    isVisionStreaming = false;
                    logError("启动 MJPEG 流失败: " + e.getMessage());
                    // Attempt to clean up camera resource if streaming fails?
                    if (webcam != null) {
                        webcam.closeCameraDeviceAsync(null); // Attempt cleanup on error
                    }
                }
            }

            @Override
            public void onError(int errorCode) {
                isVisionStreaming = false;
                logError("摄像头错误: " + errorCode);
            }
        });
    }

    public void stopVisionStream() {
        if (webcam != null && isVisionStreaming) {
            try {
                logInfo("正在停止视觉流...");
                webcam.stopStreaming();
            } catch (Exception e) {
                logError("停止流时出错: " + e.getMessage());
            }
            // Consider adding webcam.closeCameraDeviceAsync here if you want to fully release the camera
        } else if (webcam != null) {
            logInfo("摄像头存在但未在流式传输。");
        }
        isVisionStreaming = false; // Ensure flag is false even if stopping failed
    }

    public void cleanup() {
        logInfo("正在清理 RedTaskCApi...");
        stopVisionStream(); // Stop stream first

        // Now attempt to close camera device if webcam object exists
        if (webcam != null) {
            logInfo("正在关闭摄像头设备...");
            // Closing should be asynchronous to avoid blocking OpMode stop
            webcam.closeCameraDeviceAsync(new OpenCvCamera.AsyncCameraCloseListener() {
                @Override
                public void onClose() {
                    logInfo("摄像头设备已关闭。");
                }
            });
            webcam = null; // Nullify the reference
        }


        if (hSlideL != null) {
            hSlideL.setPower(0.0); // Ensure motor is stopped
            hSlideL = null; // Release reference
        }
        // Nullify servo references (hardware map handles actual device closing)
        inFlip = null;
        inFlipTurn = null;
        clawRotate = null;
        inClaw = null;

        isInitialized = false;
        isActionSequenceActive = false;
        currentActionState = ActionState.IDLE;
        hardwareMap = null; // Release hardware map reference
        telemetry = null;   // Release telemetry reference
        logInfo("清理完成。");
    }

    public void triggerActionSequence() {
        if (!isInitialized) {
            logError("无法触发动作：未初始化。");
            return;
        }
        if (isActionSequenceActive) {
            logInfo("动作序列已在运行。");
            return;
        }
        if (hSlideL == null) {
            logError("无法触发动作：滑轨电机不可用。");
            return;
        }

        // MODIFIED: Log initial speed for stage 1
        logInfo(String.format("触发动作序列：开始滑轨移动 (初始速度 %.2f)。", SLIDE_MOVE_SPEED_INITIAL));
        isActionSequenceActive = true;
        currentActionState = ActionState.MOVING_SLIDE;
        slideMoveTimer.reset();
        stateTimer.reset(); // Reset state timer as well
        // Ensure correct mode before setting power
        hSlideL.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
        // MODIFIED: Set initial speed for stage 1
        hSlideL.setPower(SLIDE_MOVE_SPEED_INITIAL);
    }

    public void update() {
        if (!isInitialized) return;

        // Always update servo targets based on vision, even if sequence isn't active
        // This allows pre-aiming or continuous adjustment if needed elsewhere
        updateServoTargetsFromVision();

        // Only run the state machine if the action sequence is active
        if (!isActionSequenceActive) {
            // Ensure state is IDLE if sequence is not active
            if (currentActionState != ActionState.IDLE) {
                transitionToActionState(ActionState.IDLE); // This also ensures motor power is 0
            }
            return; // Exit update if sequence not running
        }

        // --- Action Sequence State Machine ---
        switch (currentActionState) {
            case MOVING_SLIDE:
                handleMovingSlideState();
                break;
            case UPDATING_TURN_ROTATE:
                handleUpdatingTurnRotateState();
                break;
            case SETTING_FLIP_UP:
                handleSettingFlipUpState();
                break;
            case CLOSING_CLAW:
                handleClosingClawState();
                break;
            case SETTING_FLIP_MID:
                handleSettingFlipMidState();
                break;
            case IDLE:
                // This case should ideally not be reached if isActionSequenceActive is true,
                // but as a safeguard, ensure the sequence flag is reset.
                isActionSequenceActive = false;
                // Ensure motor is stopped if we somehow end up here while active
                if (hSlideL != null && hSlideL.getPower() != 0.0) {
                    hSlideL.setPower(0.0);
                    logInfo("警告：在IDLE状态下发现活动序列标志，停止滑轨。");
                }
                break;
        }
    }

    // MODIFIED: Implements 3-stage slide movement logic
    @SuppressLint("DefaultLocale")
    private void handleMovingSlideState() {
        if (hSlideL == null) {
            logError("滑轨移动状态期间滑轨电机丢失。正在中止。");
            transitionToActionState(ActionState.IDLE);
            return;
        }

        boolean cubeInRing = isCubeInRing();
        int currentPosition = 0;
        try {
            // It's good practice to read encoder value only once per loop iteration
            currentPosition = hSlideL.getCurrentPosition();
        } catch (Exception e) {
            logError("读取滑轨位置时出错: " + e.getMessage() + "。当前位置设为0。");
            // Potentially stop the motor or use last known good value if error persists
            // For now, we proceed assuming 0, which might cause issues if error is transient
            currentPosition = 0; // Or handle more gracefully
        }
        // double currentPower = hSlideL.getPower(); // Useful for debugging

        // --- Stopping Conditions (Checked First) ---
        if (cubeInRing) {
            logInfo(String.format("滑轨中断：在位置 %d 检测到方块。停止滑轨。", currentPosition));
            hSlideL.setPower(0.0);
            transitionToActionState(ActionState.UPDATING_TURN_ROTATE);
            return; // Exit immediately after transitioning
        }

        if (slideMoveTimer.seconds() >= SLIDE_EXTEND_TIMEOUT_SECONDS) {
            logInfo(String.format("滑轨超时：伸出 %.1f 秒后 (位置 %d) 仍未完成。正在停止并退出序列。",
                    slideMoveTimer.seconds(), currentPosition));
            hSlideL.setPower(0.0);
            transitionToActionState(ActionState.IDLE); // Exit sequence on timeout
            return; // Exit immediately after transitioning
        }

        // Check if final target position reached (and is valid)
        if (SLIDE_FINAL_TARGET_POSITION > 0 && currentPosition >= SLIDE_FINAL_TARGET_POSITION) {
            logInfo(String.format("滑轨完成：到达或超过最终目标位置 %d。停止滑轨。", SLIDE_FINAL_TARGET_POSITION));
            hSlideL.setPower(0.0);
            transitionToActionState(ActionState.UPDATING_TURN_ROTATE);
            return; // Exit immediately after transitioning
        }

        // --- Set Target Power Based on Position (Only if not stopped) ---
        double targetPower;
        if (currentPosition < SLIDE_STAGE1_TARGET_POSITION) {
            // Stage 1: 0 -> 7000
            targetPower = SLIDE_MOVE_SPEED_INITIAL;
            hSlideL.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
            hSlideL.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.FLOAT);
        } else if (currentPosition < SLIDE_INTERMEDIATE_POSITION) {
            // Stage 2: 7000 -> 19000
            targetPower = SLIDE_MOVE_SPEED_STAGE_2;
        } else {
            // Stage 3: 19000 -> 24945 (or until a stop condition is met)
            targetPower = SLIDE_MOVE_SPEED_STAGE_3;
        }

        // --- Apply Power (Optimization: Set only if changed) ---
        // Check current power to avoid redundant commands, using a small tolerance
        if (Math.abs(hSlideL.getPower() - targetPower) > 0.01) {
            hSlideL.setPower(targetPower);
            // Optional: Log power changes for debugging, but can be verbose
            // logInfo(String.format("滑轨位置 %d, 设置功率 %.2f", currentPosition, targetPower));
        }
    }


    private void handleUpdatingTurnRotateState() {
        logInfo("动作状态：更新转向/旋转伺服");
        if (inFlipTurn != null) {
            inFlipTurn.setPosition(targetInFlipTurnServoPos);
        }
        if (clawRotate != null) {
            // Apply a small adjustment if needed, or use target directly
            double finalClawRotatePos = Range.clip(targetClawRotateServoPos - 0.05, 0.0, 1.0);
            // logInfo(String.format("目标旋转位置 %.2f, 最终设置 %.2f", targetClawRotateServoPos, finalClawRotatePos)); // Debugging
            clawRotate.setPosition(finalClawRotatePos);
        }
        // Transition based on time delay to allow servos to move
        transitionToActionState(ActionState.SETTING_FLIP_UP); // Start timer for next state
    }

    private void handleSettingFlipUpState() {
        // Wait for a short duration after setting turn/rotate servos
        if (stateTimer.milliseconds() >= 200) { // 150ms delay
            logInfo("动作状态：设置翻转向上 (延迟后)");
            if (inFlip != null) {
                inFlip.setPosition(0.97); // Flip up position
            }
            transitionToActionState(ActionState.CLOSING_CLAW); // Move to next state
        }
    }

    private void handleClosingClawState() {
        // Wait for a short duration after setting flip up
        if (stateTimer.milliseconds() >= 150) { // 150ms delay
            logInfo("动作状态：闭合爪子 (延迟后)");
            if (inClaw != null) {
                inClaw.setPosition(0.75); // Close claw position
            }
            transitionToActionState(ActionState.SETTING_FLIP_MID); // Move to next state
        }
    }

    private void handleSettingFlipMidState() {
        // Wait for a short duration after closing claw
        if (stateTimer.milliseconds() >= 150) { // 150ms delay
            logInfo("动作状态：设置翻转到中间 (延迟后)");
            if (inFlip != null) {
                inFlip.setPosition(IN_FLIP_INIT_POS); // Set flip back to mid/initial position
            }

            logInfo("抓取序列完成。");
            // No need to stop slide motor here, it should already be stopped from handleMovingSlideState
            transitionToActionState(ActionState.IDLE); // Sequence complete, go to IDLE
        }
    }


    private void transitionToActionState(ActionState nextState) {
        if (currentActionState == nextState) return; // Avoid redundant transitions

        logInfo("状态转换: " + currentActionState + " -> " + nextState);
        currentActionState = nextState;
        stateTimer.reset(); // Reset timer for the *new* state's delays (if any)

        // Special handling for entering IDLE state
        if (nextState == ActionState.IDLE) {
            isActionSequenceActive = false; // Mark sequence as complete
            // Ensure motor is stopped when returning to idle, regardless of how we got here
            if (hSlideL != null && hSlideL.getPower() != 0.0) {
                hSlideL.setPower(0.0);
                logInfo("返回 IDLE 状态，确认滑轨电机已停止。");
            }
        }
    }

    // Calculates target servo positions based on vision data
    @SuppressLint("DefaultLocale")
    private void updateServoTargetsFromVision() {
        if (BLUEDetectionPipeline == null || !isVisionStreaming) {
            // Do not update targets if vision is not ready
            // Consider setting to default/init positions if vision lost? Optional.
            return;
        }

        DetectedCube currentSelectedCube = BLUEDetectionPipeline.getSelectedCube();

        if (currentSelectedCube != null) {
            double inputTurnAngle = currentSelectedCube.angleFromVertical;
            double calculatedTurnPos = ((0.7 / 180) * (inputTurnAngle + 90)) + 0.01;
            targetInFlipTurnServoPos = Range.clip(calculatedTurnPos, 0, 1);
            double adjustedRotateAngle = currentSelectedCube.objectOrientationDeg - inputTurnAngle; // Angle relative to the grabber's approach
            int FinalRotateAngle = (int) Math.abs(adjustedRotateAngle);
            double originalCalcRotatePos = 1 - (((0.68/180) * (FinalRotateAngle)) + 0.29);

            targetClawRotateServoPos = Range.clip(originalCalcRotatePos, 0.0, 1.0);

        }
    }


    // Checks if the currently selected cube (if any) is marked as being in the ring
    public boolean isCubeInRing() {
        if (BLUEDetectionPipeline == null) return false;
        DetectedCube cube = BLUEDetectionPipeline.getSelectedCube();
        // The check relies solely on the flag set within the pipeline's processing logic
        return (cube != null && cube.isInRing);
    }

    // --- Getters ---
    public DetectedCube getSelectedCubeData() {
        // Returns a *copy* of the selected cube data (thread-safe)
        return (BLUEDetectionPipeline != null) ? BLUEDetectionPipeline.getSelectedCube() : null;
    }

    public boolean isVisionStreamActive() {
        return isVisionStreaming;
    }

    public boolean isActionSequenceRunning() {
        return isActionSequenceActive;
    }

    public ActionState getCurrentActionState() {
        return currentActionState;
    }

    // --- Logging ---
    private void logInfo(String message) {
        if (telemetry != null) {
            telemetry.addLine("RedTaskCApi: " + message);
            telemetry.update(); // Consider updating less frequently if performance is an issue
        }
        System.out.println("RedTaskCApi INFO: " + message); // Log to console/logcat
    }

    private void logError(String message) {
        if (telemetry != null) {
            telemetry.addLine("!!! RedTaskCApi 错误: " + message);
            telemetry.update(); // Update immediately for errors
        }
        System.err.println("!!! RedTaskCApi 错误: " + message); // Log to error stream
    }

    // MODIFIED: Updated printConfig for 3-stage slide
    @SuppressLint("DefaultLocale")
    private void printConfig() {
        if (telemetry == null) return;
        // Use telemetry.log() for persistent messages if desired, or telemetry.addData for per-loop updates
        telemetry.log().clear(); // Clear previous persistent logs
        telemetry.log().add("===================== RedTaskCApi 配置 =====================");
        telemetry.log().add("--- 视觉 ---");
        telemetry.log().add(String.format(" 摄像头: %s (%dx%d, %s)", WEBCAM_NAME, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_STREAMING_ROTATION));
        telemetry.log().add(String.format(" 图像旋转180度: %s, 缩小因子: %.2f", IMAGE_ROTATE_180, DOWNSCALE_FACTOR));
        telemetry.log().add(String.format(" 像素/厘米比例 (cm/px): %.5f", PIXELS_TO_CM_RATIO));
        telemetry.log().add(String.format(" 环中心偏移 (cm): X=%.1f, Y=%.1f", CIRCLE_CENTER_OFFSET_CM_X, CIRCLE_CENTER_OFFSET_CM_Y));
        telemetry.log().add(String.format(" 环半径 (cm): 内=%.1f, 宽度=%.1f", ARM_BACK_CM, ERROR_CM));
        telemetry.log().add(String.format(" 可抓取角度: %.1f 度", GRASPABLE_ANGLE_DEG));
        telemetry.log().add(String.format(" 尺寸过滤 (像素): %.0f - %.0f", MIN_SIZE_PIXELS, MAX_SIZE_PIXELS));
        telemetry.log().add("--- 伺服 ---");
        telemetry.log().add(String.format(" 初始位置: Flip=%.2f, Turn=%.2f, Rotate=%.2f, Claw=%.2f", IN_FLIP_INIT_POS, IN_FLIP_TURN_INIT_POS, CLAW_ROTATE_INIT_POS, IN_CLAW_INIT_POS));
        telemetry.log().add(String.format(" 转向映射: [%.1f, %.1f]度 -> [%.2f, %.2f]位置", -90.0, 90.0, SERVO_TURN_MIN_POS, SERVO_TURN_MAX_POS)); // Updated angle range display
        telemetry.log().add(String.format(" 旋转映射: [%.2f, %.2f]位置 (约 0-180度相对角, 反向)", SERVO_ROTATE_MIN_POS, SERVO_ROTATE_MAX_POS)); // Updated description
        telemetry.log().add("--- 滑轨 ---");
        telemetry.log().add(String.format(" 电机: %s (反向, 刹车, RunWithoutEncoder)", SLIDE_MOTOR_NAME));
        // Updated slide stage descriptions
        telemetry.log().add(String.format(" 伸出阶段 1: -> %d ticks @ %.2f power", SLIDE_STAGE1_TARGET_POSITION, SLIDE_MOVE_SPEED_INITIAL));
        telemetry.log().add(String.format(" 伸出阶段 2: %d -> %d ticks @ %.2f power", SLIDE_STAGE1_TARGET_POSITION, SLIDE_INTERMEDIATE_POSITION, SLIDE_MOVE_SPEED_STAGE_2));
        telemetry.log().add(String.format(" 伸出阶段 3: %d -> %d ticks @ %.2f power", SLIDE_INTERMEDIATE_POSITION, SLIDE_FINAL_TARGET_POSITION, SLIDE_MOVE_SPEED_STAGE_3));
        telemetry.log().add(String.format(" 伸出超时: %.1f 秒 (超时将停止并退出)", SLIDE_EXTEND_TIMEOUT_SECONDS));
        telemetry.log().add("---------------------------------------------------------------");
        telemetry.update(); // Show config log on telemetry screen
    }


    // --- Inner Classes (BLUEDetectionPipeline, DetectedCube) ---
    // Assume these are unchanged from the original provided code unless modifications
    // related to the main class logic were needed (none in this case).
    // Make sure to copy the *exact* code for these inner classes from your working version.
    public static class BLUEDetectionPipeline extends OpenCvPipeline {
        // --- Constants copied from RedTaskCApi ---
        // Ensure these match the outer class's constants *exactly*
        private static final boolean IMAGE_ROTATE_180 = RedTaskCApi.IMAGE_ROTATE_180;
        private static final double DOWNSCALE_FACTOR = RedTaskCApi.DOWNSCALE_FACTOR;
        private static final double ARM_BACK_CM = RedTaskCApi.ARM_BACK_CM;
        private static final double ERROR_CM = RedTaskCApi.ERROR_CM;
        private static final double CIRCLE_CENTER_OFFSET_CM_X = RedTaskCApi.CIRCLE_CENTER_OFFSET_CM_X;
        private static final double CIRCLE_CENTER_OFFSET_CM_Y = RedTaskCApi.CIRCLE_CENTER_OFFSET_CM_Y;
        private static final double GRASPABLE_ANGLE_DEG = RedTaskCApi.GRASPABLE_ANGLE_DEG;
        private static final double PIXELS_TO_CM_RATIO = RedTaskCApi.PIXELS_TO_CM_RATIO;
        private static final double MIN_SIZE_PIXELS = RedTaskCApi.MIN_SIZE_PIXELS;
        private static final double MAX_SIZE_PIXELS = RedTaskCApi.MAX_SIZE_PIXELS;
        private static final List<Scalar[]> BLUE_HSV_RANGES = RedTaskCApi.RED_HSV_RANGES; // Name adjusted for clarity, uses RED ranges
        private static final Scalar CIRCLE_COLOR = RedTaskCApi.CIRCLE_COLOR;
        private static final int CIRCLE_THICKNESS = RedTaskCApi.CIRCLE_THICKNESS;
        private static final Scalar PERSISTENT_LINE_COLOR = RedTaskCApi.PERSISTENT_LINE_COLOR;
        private static final int PERSISTENT_LINE_THICKNESS = RedTaskCApi.PERSISTENT_LINE_THICKNESS;
        private static final Scalar BOX_COLOR_SELECTED = RedTaskCApi.BOX_COLOR_SELECTED;
        private static final Scalar BOX_COLOR_DEFAULT = RedTaskCApi.BOX_COLOR_DEFAULT;
        private static final Scalar CENTER_TO_TARGET_LINE_COLOR = RedTaskCApi.CENTER_TO_TARGET_LINE_COLOR;
        private static final int CONNECTION_LINE_THICKNESS = RedTaskCApi.CONNECTION_LINE_THICKNESS;
        private static final Scalar BLOCK_CENTER_CIRCLE_COLOR = RedTaskCApi.BLOCK_CENTER_CIRCLE_COLOR;
        private static final Scalar TEXT_COLOR = RedTaskCApi.TEXT_COLOR;
        private static final Scalar GRASPABLE_ANGLE_LINE_COLOR = RedTaskCApi.GRASPABLE_ANGLE_LINE_COLOR;
        private static final int GRASPABLE_ANGLE_LINE_THICKNESS = RedTaskCApi.GRASPABLE_ANGLE_LINE_THICKNESS;

        // --- Pipeline Members ---
        private final Mat rgbImage = new Mat();
        private final Mat hsvImage = new Mat();
        private final Mat BLUEMask = new Mat(); // Renamed for clarity, still uses RED ranges
        private final Mat rangeMask = new Mat();
        private final Mat hierarchy = new Mat();
        private final MatOfPoint2f contour2f = new MatOfPoint2f(); // Reusable buffer
        private final Lock detectedCubesLock = new ReentrantLock(); // For thread safety
        private List<DetectedCube> allDetectedCubes = new ArrayList<>(); // Cubes from current frame
        private volatile DetectedCube selectedCube = null; // The chosen cube (volatile for visibility)
        private volatile boolean viewportPaused = false;

        // Public getter for selected cube (thread-safe)
        public DetectedCube getSelectedCube() {
            detectedCubesLock.lock();
            try {
                // Return a *copy* to prevent external modification of the pipeline's state
                return (selectedCube == null) ? null : new DetectedCube(selectedCube);
            } finally {
                detectedCubesLock.unlock();
            }
        }

        @Override
        public Mat processFrame(Mat input) {
            // Handle viewport pause state
            if (viewportPaused) {
                // Optionally draw pause indicator if needed
                // Imgproc.putText(input, "PAUSED", ...);
                return input; // Return input directly without processing
            }

            Mat displayOutput = input; // Default to drawing on the input buffer

            // 1. Rotation (if configured)
            if (IMAGE_ROTATE_180) {
                Core.rotate(input, input, Core.ROTATE_180);
                // Input buffer is now rotated, displayOutput points to it
            }

            // 2. Downscaling (if configured) - Process smaller, draw on original/rotated
            Mat processedInput; // The image actually used for detection
            double drawScale = 1.0; // Scale factor for drawing back onto displayOutput
            if (DOWNSCALE_FACTOR < 1.0 && DOWNSCALE_FACTOR > 0) {
                // Calculate scaled dimensions
                int scaledWidth = (int)(input.cols() * DOWNSCALE_FACTOR);
                int scaledHeight = (int)(input.rows() * DOWNSCALE_FACTOR);
                // Resize input into rgbImage (reused buffer)
                Imgproc.resize(input, rgbImage, new Size(scaledWidth, scaledHeight), 0, 0, Imgproc.INTER_LINEAR);
                processedInput = rgbImage; // Process the downscaled image
                drawScale = 1.0 / DOWNSCALE_FACTOR; // Calculate scale for drawing
            } else {
                // No downscaling, process directly on input (or a copy)
                // Ensure 3 channels (RGB) if input is RGBA
                if(input.channels() == 4) {
                    Imgproc.cvtColor(input, rgbImage, Imgproc.COLOR_RGBA2RGB);
                    processedInput = rgbImage; // Process the 3-channel copy
                } else {
                    // Input is already RGB or Grayscale, process it directly (or copy)
                    // Using input directly might modify it if later steps do in-place ops
                    // input.copyTo(rgbImage); // Safer: work on a copy
                    // processedInput = rgbImage;
                    processedInput = input; // Process in-place (careful!)
                }
                drawScale = 1.0; // No scaling needed for drawing
            }

            // Get dimensions of the image being processed
            int height = processedInput.rows();
            int width = processedInput.cols();
            if (height <= 0 || width <= 0) {
                System.err.println("Pipeline Error: Invalid processed image dimensions.");
                return displayOutput; // Return the display buffer without further processing
            }

            // Calculate operational parameters in *processed* image pixels
            int armBackPx = toPx(ARM_BACK_CM);
            int errorPx = toPx(ERROR_CM);
            int armForwardPx = armBackPx + errorPx;
            int circleCenterOffsetPxX = toPx(CIRCLE_CENTER_OFFSET_CM_X);
            int circleCenterOffsetPxY = toPx(CIRCLE_CENTER_OFFSET_CM_Y);
            Point operationalCenterPx = new Point(
                    (double)width / 2.0 + circleCenterOffsetPxX,
                    (double)height / 2.0 + circleCenterOffsetPxY
            );
            // Use a large angle if GRASPABLE_ANGLE_DEG is invalid/disabled
            double halfGraspAngle = (0 < GRASPABLE_ANGLE_DEG && GRASPABLE_ANGLE_DEG < 360)
                    ? GRASPABLE_ANGLE_DEG / 2.0 : 181.0; // 181 effectively disables angle check

            // 3. Color Conversion (RGB -> HSV)
            Imgproc.cvtColor(processedInput, hsvImage, Imgproc.COLOR_RGB2HSV);

            // 4. HSV Thresholding (Combine multiple ranges)
            // Ensure mask buffers are allocated and match HSV image size
            if (BLUEMask.empty() || BLUEMask.size().equals(hsvImage.size())) {
                BLUEMask.create(hsvImage.size(), CvType.CV_8UC1);
            }
            if (rangeMask.empty() || rangeMask.size().equals(hsvImage.size())) {
                rangeMask.create(hsvImage.size(), CvType.CV_8UC1);
            }
            BLUEMask.setTo(Scalar.all(0)); // Clear the combined mask for this frame

            for (Scalar[] range : BLUE_HSV_RANGES) { // Iterate through defined HSV ranges
                if (range != null && range.length == 2 && range[0] != null && range[1] != null) {
                    Core.inRange(hsvImage, range[0], range[1], rangeMask); // Threshold for current range
                    Core.bitwise_or(BLUEMask, rangeMask, BLUEMask); // Combine with main mask
                }
            }

            // 5. Find Contours
            List<MatOfPoint> contours = new ArrayList<>();
            // Release/recreate hierarchy to avoid issues from previous frames
            if (hierarchy != null && !hierarchy.empty()) hierarchy.release();
            hierarchy.create(0, 0, CvType.CV_32SC4); // Create empty hierarchy mat
            Imgproc.findContours(BLUEMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            // 6. Process Contours
            List<DetectedCube> currentFrameCubes = new ArrayList<>();
            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);

                // Filter by area
                if (area >= MIN_SIZE_PIXELS && area <= MAX_SIZE_PIXELS) {
                    Rect br = Imgproc.boundingRect(contour);
                    Point cp = new Point(br.x + br.width / 2.0, br.y + br.height / 2.0); // Center point

                    // Calculate orientation using minAreaRect (more robust than ellipse for rectangles)
                    double objectOrientationDeg = 0.0;
                    if (contour.total() >= 5) { // Need at least 5 points
                        // Convert contour to MatOfPoint2f for minAreaRect
                        contour.convertTo(contour2f, CvType.CV_32F);
                        try {
                            RotatedRect rr = Imgproc.minAreaRect(contour2f);
                            // Angle is [-90, 0). Adjust based on aspect ratio.
                            // Angle of the longer side relative to horizontal axis.
                            if (rr.size.width < rr.size.height) {
                                objectOrientationDeg = rr.angle + 90.0; // Vertical object
                            } else {
                                objectOrientationDeg = rr.angle; // Horizontal object
                            }
                            // Normalize to [0, 180) or keep as is depending on need
                            // objectOrientationDeg = (objectOrientationDeg + 180) % 180; // Example normalization
                        } catch (CvException e) {
                            System.err.println("Error calculating minAreaRect: " + e.getMessage());
                            // Orientation remains 0.0
                        } finally {
                            // contour2f is reused, no need to release here if pre-allocated member
                        }
                    }

                    // Calculate distance squared from operational center
                    double dx = cp.x - operationalCenterPx.x;
                    double dy = cp.y - operationalCenterPx.y;
                    double distSq = dx * dx + dy * dy;
                    boolean isInRing = (distSq >= (double)armBackPx * armBackPx && distSq <= (double)armForwardPx * armForwardPx);

                    // Calculate angle relative to the vertical line through the operational center
                    double angleFromVertical = calculateAngleFromVertical(cp, operationalCenterPx); // -180 to +180, 0 is up
                    boolean isInGraspAngle = Math.abs(angleFromVertical) <= halfGraspAngle;

                    // Add detected cube information (using processed image coords/scale)
                    currentFrameCubes.add(new DetectedCube("RED", // Color label (using RED ranges)
                            (int) Math.round(cp.x), (int) Math.round(cp.y),
                            objectOrientationDeg, br, area, isInRing, angleFromVertical, isInGraspAngle,
                            DOWNSCALE_FACTOR)); // Store the scale factor used
                }
                contour.release(); // Release contour MatOfPoint memory
            }

            // 7. Select the Best Cube based on criteria
            DetectedCube currentSelectedCube = null;
            List<DetectedCube> candidates = new ArrayList<>();
            // Primary criteria: In Ring AND In Graspable Angle
            for (DetectedCube cube : currentFrameCubes) {
                if (cube.isInRing && cube.isInGraspAngle) {
                    candidates.add(cube);
                }
            }

            if (!candidates.isEmpty()) {
                // If candidates found, sort by closeness to center line (smallest absolute angle)
                candidates.sort(Comparator.comparingDouble(c -> Math.abs(c.angleFromVertical)));
                currentSelectedCube = candidates.get(0);
            } else {
                // Fallback 1: If none in ring AND angle, check for just In Ring
                candidates.clear();
                for (DetectedCube cube : currentFrameCubes) {
                    if (cube.isInRing) {
                        candidates.add(cube);
                    }
                }
                if (!candidates.isEmpty()) {
                    // Sort those in ring by angle and pick closest
                    candidates.sort(Comparator.comparingDouble(c -> Math.abs(c.angleFromVertical)));
                    currentSelectedCube = candidates.get(0);
                }
                // Fallback 2 (Optional): If still no cube, maybe pick the largest / closest overall?
                // else if (!currentFrameCubes.isEmpty()) {
                //     currentFrameCubes.sort(Comparator.comparingDouble(DetectedCube::getAreaPx).reversed()); // Largest
                //     currentSelectedCube = currentFrameCubes.get(0);
                // }
            }

            // 8. Update Shared State (Thread-Safe)
            detectedCubesLock.lock();
            try {
                // Store a *copy* of the selected cube (or null)
                this.selectedCube = (currentSelectedCube == null) ? null : new DetectedCube(currentSelectedCube);
                // Update the list of all cubes detected in this frame (optional, for debugging/display)
                this.allDetectedCubes = currentFrameCubes; // Can assign directly if currentFrameCubes is local
            } finally {
                detectedCubesLock.unlock();
            }

            // 9. Draw Overlays onto the *displayOutput* buffer (original size or rotated)
            // Calculate drawing parameters based on displayOutput dimensions and drawScale
            int drawWidth = displayOutput.cols();
            int drawHeight = displayOutput.rows();
            // Use a helper to calculate pixel values from CM using the *drawing* scale
            double drawPixelRatio = (PIXELS_TO_CM_RATIO <= 1e-9) ? 0 : PIXELS_TO_CM_RATIO / drawScale;
            Point drawOperationalCenterPx = new Point(
                    (double)drawWidth / 2.0 + toPxCm(CIRCLE_CENTER_OFFSET_CM_X, drawPixelRatio),
                    (double)drawHeight / 2.0 + toPxCm(CIRCLE_CENTER_OFFSET_CM_Y, drawPixelRatio)
            );
            int drawArmBackPx = toPxCm(ARM_BACK_CM, drawPixelRatio);
            int drawErrorPx = toPxCm(ERROR_CM, drawPixelRatio);
            int drawArmForwardPx = drawArmBackPx + drawErrorPx;

            // Draw static elements (ring arcs, lines)
            drawStaticOverlays(displayOutput, drawOperationalCenterPx, drawArmBackPx, drawArmForwardPx, halfGraspAngle);
            // Draw detected cubes (boxes, centers, info) - use `allDetectedCubes` from shared state for safety
            detectedCubesLock.lock();
            try{
                drawDetections(displayOutput, this.allDetectedCubes, this.selectedCube, drawOperationalCenterPx, drawScale);
            } finally {
                detectedCubesLock.unlock();
            }


            // 10. Release Intermediate Mats (if they were copies, not strictly necessary for member fields if reused)
            // hsvImage, BLUEMask, rangeMask, hierarchy are member fields, reused.
            // rgbImage is reused or points to input.
            // contour2f is reused.
            // contours list elements were released individually.

            return displayOutput; // Return the image with overlays drawn
        }

        // --- Drawing Helper Methods ---

        private void drawStaticOverlays(Mat image, Point center, int radiusBack, int radiusFwd, double halfAngleDeg) {
            int h = image.rows();
            int w = image.cols();
            int cx = (int) Math.round(center.x);
            int cy = (int) Math.round(center.y);

            // Draw vertical line through operational center
            Imgproc.line(image, new Point(cx, 0), new Point(cx, h), PERSISTENT_LINE_COLOR, PERSISTENT_LINE_THICKNESS);

            // Draw graspable area arcs and lines
            boolean drawArcs = (halfAngleDeg > 0 && halfAngleDeg <= 180.0 && radiusBack > 0 && radiusFwd > 0 && radiusFwd > radiusBack);
            if (drawArcs) {
                // Angles for ellipse: 0 is right, 90 up, 180 left, 270 down.
                // Vertical center line corresponds to 270 degrees.
                double startAngle = 270.0 - halfAngleDeg; // Angle towards left
                double endAngle = 270.0 + halfAngleDeg;   // Angle towards right
                Size axesBack = new Size(radiusBack, radiusBack);
                Size axesFwd = new Size(radiusFwd, radiusFwd);

                // Draw the arcs defining the ring segment
                Imgproc.ellipse(image, center, axesBack, 0, startAngle, endAngle, CIRCLE_COLOR, CIRCLE_THICKNESS, Imgproc.LINE_AA);
                Imgproc.ellipse(image, center, axesFwd, 0, startAngle, endAngle, CIRCLE_COLOR, CIRCLE_THICKNESS, Imgproc.LINE_AA);

                // Draw the radial lines indicating the angle limits
                double angleRadLeft = Math.toRadians(startAngle);
                double angleRadRight = Math.toRadians(endAngle);
                // Extend lines slightly beyond arcs for visibility
                Point startLeft = new Point(center.x + (radiusBack - 5) * Math.cos(angleRadLeft), center.y + (radiusBack - 5) * Math.sin(angleRadLeft));
                Point startRight = new Point(center.x + (radiusBack - 5) * Math.cos(angleRadRight), center.y + (radiusBack - 5) * Math.sin(angleRadRight));
                Point endLeft = new Point(center.x + (radiusFwd + 5) * Math.cos(angleRadLeft), center.y + (radiusFwd + 5) * Math.sin(angleRadLeft));
                Point endRight = new Point(center.x + (radiusFwd + 5) * Math.cos(angleRadRight), center.y + (radiusFwd + 5) * Math.sin(angleRadRight));

                Imgproc.line(image, startLeft, endLeft, GRASPABLE_ANGLE_LINE_COLOR, GRASPABLE_ANGLE_LINE_THICKNESS, Imgproc.LINE_AA);
                Imgproc.line(image, startRight, endRight, GRASPABLE_ANGLE_LINE_COLOR, GRASPABLE_ANGLE_LINE_THICKNESS, Imgproc.LINE_AA);

            } else { // Draw full circles if angle check is disabled or radii invalid
                if (radiusBack > 0) Imgproc.circle(image, center, radiusBack, CIRCLE_COLOR, CIRCLE_THICKNESS, Imgproc.LINE_AA);
                if (radiusFwd > 0) Imgproc.circle(image, center, radiusFwd, CIRCLE_COLOR, CIRCLE_THICKNESS, Imgproc.LINE_AA);
            }
            // Draw a marker at the operational center
            Imgproc.circle(image, center, 5, new Scalar(255,0,255), -1); // Magenta center dot
        }

        // Draws detections (boxes, centers, text) onto the display image
        private void drawDetections(Mat displayOutput, List<DetectedCube> cubes, DetectedCube selected, Point drawOpCenter, double drawScale) {
            if (cubes == null) return;

            for (DetectedCube cube : cubes) {
                // Scale detection results from processed coords back to display coords
                Rect drawBox = scaleRect(cube.boundingBox, drawScale);
                Point drawCenter = scalePoint(new Point(cube.centerXImagePx, cube.centerYImagePx), drawScale);

                // Choose color based on selection and status
                Scalar boxColor;
                // Use equals() for object comparison if 'selected' might be a different instance with same data
                // But since we store a copy in 'selectedCube' member and pass that, reference check (==) is fine here.
                if (cube == selected) {
                    boxColor = BOX_COLOR_SELECTED; // Red for the selected cube
                } else if (cube.isInRing && cube.isInGraspAngle) {
                    boxColor = new Scalar(255, 165, 0); // Orange for valid candidates not selected
                } else if (cube.isInRing) {
                    boxColor = new Scalar(255, 255, 0); // Yellow for in ring but outside angle
                }
                else {
                    boxColor = BOX_COLOR_DEFAULT; // Default green for others
                }

                // Draw bounding box and center dot
                Imgproc.rectangle(displayOutput, drawBox, boxColor, 2);
                Imgproc.circle(displayOutput, drawCenter, 4, BLOCK_CENTER_CIRCLE_COLOR, -1);

                // Prepare and draw text labels
                String labelColor = cube.color;
                String labelAngleV = String.format(Locale.US, "V:%.1f", cube.angleFromVertical);
                String labelAngleO = String.format(Locale.US, "O:%.1f", cube.objectOrientationDeg);
                // Display coordinates in the *display* frame
                String labelPos = String.format(Locale.US, "P:(%d,%d)", (int)Math.round(drawCenter.x), (int)Math.round(drawCenter.y));
                String labelStatus = String.format(Locale.US,"R:%b A:%b", cube.isInRing, cube.isInGraspAngle);

                // Position text near the top-left of the bounding box
                Point textOrigin = new Point(drawBox.x, drawBox.y - 5); // Start slightly above box
                if (textOrigin.y < 15) textOrigin.y = drawBox.y + drawBox.height + 15; // Move below if too high
                double fontScale = 0.4;
                int thickness = 1;
                int lineSpacing = 15; // Pixels between lines of text

                // Draw the text lines using LINE_AA for anti-aliasing
                Imgproc.putText(displayOutput, labelColor + " " + labelStatus, textOrigin, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, TEXT_COLOR, thickness, Imgproc.LINE_AA);
                Imgproc.putText(displayOutput, labelAngleV, new Point(textOrigin.x, textOrigin.y + lineSpacing), Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, TEXT_COLOR, thickness, Imgproc.LINE_AA);
                Imgproc.putText(displayOutput, labelAngleO, new Point(textOrigin.x, textOrigin.y + 2*lineSpacing), Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, TEXT_COLOR, thickness, Imgproc.LINE_AA);
                Imgproc.putText(displayOutput, labelPos, new Point(textOrigin.x, textOrigin.y + 3*lineSpacing), Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, TEXT_COLOR, thickness, Imgproc.LINE_AA);
            }

            // Draw line from operational center to the selected cube's center (if one exists)
            if (selected != null) {
                // Scale selected cube's center to display coordinates
                Point drawSelectedCenter = scalePoint(new Point(selected.centerXImagePx, selected.centerYImagePx), drawScale);
                Imgproc.line(displayOutput, drawOpCenter, drawSelectedCenter, CENTER_TO_TARGET_LINE_COLOR, CONNECTION_LINE_THICKNESS, Imgproc.LINE_AA);
            }
        }


        // Toggles the processing pause state when viewport is tapped
        @Override
        public void onViewportTapped() {
            viewportPaused = !viewportPaused;
            System.out.println("Viewport Tapped! Pause state: " + viewportPaused);
        }

        // --- Helper Methods ---

        // Calculates angle relative to the upward vertical line through centerPx.
        // Returns angle in degrees (-180 to +180). 0 is up, positive is clockwise.
        private static double calculateAngleFromVertical(Point pointPx, Point centerPx) {
            double dx = pointPx.x - centerPx.x; // Positive is right
            double dy = centerPx.y - pointPx.y; // Positive is up (adjusting for image coords)
            double angleRad = Math.atan2(dx, dy); // atan2(x, y) gives angle from positive Y axis
            return Math.toDegrees(angleRad);
        }

        // Converts CM to pixels using the class constant ratio (for processed image)
        private static int toPx(double cm) {
            if (cm <= 0 || PIXELS_TO_CM_RATIO <= 1e-9) return 0;
            int px = (int) Math.round(cm / PIXELS_TO_CM_RATIO);
            return Math.max(1, px); // Ensure at least 1 pixel if cm > 0
        }

        // Converts CM to pixels using a *provided* ratio (useful for drawing scale)
        private static int toPxCm(double cm, double cmPerPixelRatio) {
            if (cm <= 0 || cmPerPixelRatio <= 1e-9) return 0;
            int px = (int) Math.round(cm / cmPerPixelRatio);
            return Math.max(1, px); // Ensure at least 1 pixel if cm > 0
        }

        // Scales a Point by a factor
        private Point scalePoint(Point point, double scale) {
            if (Math.abs(scale - 1.0) < 1e-3) return point.clone(); // Avoid scaling if factor is ~1
            return new Point(point.x * scale, point.y * scale);
        }

        // Scales a Rect by a factor
        private Rect scaleRect(Rect rect, double scale) {
            if (Math.abs(scale - 1.0) < 1e-3) return rect.clone(); // Avoid scaling if factor is ~1
            return new Rect(
                    (int) Math.round(rect.x * scale),
                    (int) Math.round(rect.y * scale),
                    (int) Math.round(rect.width * scale),
                    (int) Math.round(rect.height * scale)
            );
        }

        // --- Resource Management ---
        // Called when the pipeline is removed or the camera closed. Release Mats here.
        public void releaseMats() {
            System.out.println("Releasing pipeline Mats...");
            if (rgbImage != null) rgbImage.release();
            if (hsvImage != null) hsvImage.release();
            if (BLUEMask != null) BLUEMask.release();
            if (rangeMask != null) rangeMask.release();
            if (hierarchy != null) hierarchy.release();
            if (contour2f != null) contour2f.release(); // Release reusable buffer
            System.out.println("Finished releasing pipeline Mats.");
        }

        // Optional: Use finalize as a backup, but explicit release is better
        @Override
        protected void finalize() throws Throwable {
            try {
                releaseMats();
            } finally {
                super.finalize();
            }
        }

    } // End of BLUEDetectionPipeline class


    // Data class to hold information about a detected cube
    public static class DetectedCube {
        final String color;             // e.g., "RED"
        final int centerXImagePx;       // Center X in *processed* image coordinates
        final int centerYImagePx;       // Center Y in *processed* image coordinates
        final double objectOrientationDeg; // Orientation angle in degrees
        final Rect boundingBox;         // Bounding box in *processed* image coordinates
        final double areaPx;            // Area in pixels^2 in *processed* image
        final boolean isInRing;         // Is the center within the target ring?
        final double angleFromVertical; // Angle from vertical line (-180 to 180)
        final boolean isInGraspAngle;   // Is the angle within the graspable range?
        final double scaleFactor;       // The DOWNSCALE_FACTOR used during detection

        // Constructor
        public DetectedCube(String c, int x, int y, double oa, Rect r, double a, boolean ir, double av, boolean ia, double scale) {
            this.color = c;
            this.centerXImagePx = x;
            this.centerYImagePx = y;
            this.objectOrientationDeg = oa;
            this.boundingBox = r.clone(); // Clone Rect for immutability
            this.areaPx = a;
            this.isInRing = ir;
            this.angleFromVertical = av;
            this.isInGraspAngle = ia;
            this.scaleFactor = scale;
        }

        // Copy constructor - essential for thread safety when passing data out
        public DetectedCube(DetectedCube other) {
            this(other.color, other.centerXImagePx, other.centerYImagePx, other.objectOrientationDeg,
                    other.boundingBox, // Already cloned in primary constructor
                    other.areaPx, other.isInRing, other.angleFromVertical, other.isInGraspAngle, other.scaleFactor);
        }

        // --- Helper methods to get values in *original* image coordinates ---

        // Get center point scaled back to original image coordinates
        public Point getCenterInOriginalCoords() {
            if (scaleFactor <= 1e-9 || Math.abs(scaleFactor - 1.0) < 1e-3) {
                return new Point(centerXImagePx, centerYImagePx); // No scaling needed
            }
            return new Point(Math.round(centerXImagePx / scaleFactor), Math.round(centerYImagePx / scaleFactor));
        }

        // Get bounding box scaled back to original image coordinates
        public Rect getBoundingBoxInOriginalCoords() {
            if (scaleFactor <= 1e-9 || Math.abs(scaleFactor - 1.0) < 1e-3) {
                return boundingBox.clone(); // No scaling needed
            }
            return new Rect(
                    (int) Math.round(boundingBox.x / scaleFactor),
                    (int) Math.round(boundingBox.y / scaleFactor),
                    (int) Math.round(boundingBox.width / scaleFactor),
                    (int) Math.round(boundingBox.height / scaleFactor)
            );
        }

        // Get area scaled back to original image pixel^2
        public double getAreaInOriginalCoords() {
            if (scaleFactor <= 1e-9 || Math.abs(scaleFactor - 1.0) < 1e-3) {
                return areaPx; // No scaling needed
            }
            // Area scales with the square of the linear scale factor
            return areaPx / (scaleFactor * scaleFactor);
        }

        @Override
        public String toString() {
            return String.format(Locale.US, "Cube[Color:%s, Center:(%d,%d)@%.2fx, Area:%.1f, Orient:%.1f, Ring:%b, AngleV:%.1f, Grasp:%b]",
                    color, centerXImagePx, centerYImagePx, scaleFactor, areaPx, objectOrientationDeg, isInRing, angleFromVertical, isInGraspAngle);
        }

    } // End of DetectedCube class

} // End of RedTaskCApi class