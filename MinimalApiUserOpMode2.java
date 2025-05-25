package opmode.auto;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;

@TeleOp(name = "Minimal API BLUE", group = "Examples")
public class MinimalApiUserOpMode2 extends LinearOpMode {

    private BlueTaskCApi blueTaskCApi;

    /**private RedTaskCApi redTaskCApi;*/

    @Override
    public void runOpMode() throws InterruptedException {

        blueTaskCApi = new BlueTaskCApi();
        boolean initOk = blueTaskCApi.initialize(hardwareMap, null);
        if (!initOk) {
            requestOpModeStop();
            return;
        }
        blueTaskCApi.startVisionStream();
        waitForStart();

        while (opModeIsActive()) {

            blueTaskCApi.update();
            if (gamepad1.right_bumper && !blueTaskCApi.isActionSequenceRunning()) {
                blueTaskCApi.triggerActionSequence(); // Call the API method
            }
        }
        blueTaskCApi.cleanup();
    }
}