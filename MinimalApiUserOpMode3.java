package opmode.auto;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;

import hardware.RobotHardware;

@TeleOp(name = "Minimal API RED", group = "Examples")
public class MinimalApiUserOpMode3 extends LinearOpMode {



    private RedTaskCApi redTaskCApi;
    private RobotHardware poseidon;


    @Override
    public void runOpMode() throws InterruptedException {

        redTaskCApi = new RedTaskCApi();
        boolean initOK = redTaskCApi.initialize(hardwareMap, null);
        if (!initOK) {
            requestOpModeStop();
            return;
        }
        redTaskCApi.startVisionStream();
        poseidon = new RobotHardware(hardwareMap);
        waitForStart();

        while (opModeIsActive()) {
            redTaskCApi.update();
            if (gamepad1.right_bumper && !redTaskCApi.isActionSequenceRunning()) {
                poseidon.inFlip.setPosition(0.7);
                poseidon.inFlipTurn.setPosition(0.38);
                poseidon.clawRotate.setPosition(0.29);
                poseidon.inClaw.setPosition(0.34);

                sleep(100);
                redTaskCApi.triggerActionSequence();
            }
        }
        redTaskCApi.cleanup();
    }
}