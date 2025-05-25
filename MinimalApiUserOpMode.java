package opmode.auto;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;

@TeleOp(name = "Minimal API YELLOW", group = "Examples")
public class MinimalApiUserOpMode extends LinearOpMode {

    /**private BlueTaskCApi blueTaskCApi;*/

    /**private RedTaskCApi redTaskCApi;*/
    private TaskBApi taskBApi;

    @Override
    public void runOpMode() throws InterruptedException {

        /**用于仅初始化红色
        redTaskCApi = new RedTaskCApi();
        boolean initOK = redTaskCApi.initialize(hardwareMap, null);
        if (!initOK) {
            requestOpModeStop();
            return;
        }
        redTaskCApi.startVisionStream();
        waitForStart();*/

        /** 用于仅初始化蓝色
        blueTaskCApi = new BlueTaskCApi();
        boolean initOk = blueTaskCApi.initialize(hardwareMap, null);
        if (!initOk) {
            requestOpModeStop();
            return;
        }
        blueTaskCApi.startVisionStream();
        waitForStart();*/

        /**用于仅初始化黄色*/
        taskBApi = new TaskBApi();
        boolean initOK = taskBApi.initialize(hardwareMap, null);
        if (!initOK) {
            requestOpModeStop();
            return;
        }
        taskBApi.startVisionStream();
        waitForStart();

        while (opModeIsActive()) {
            /**用于仅抓取红色
            redTaskCApi.update();
            if (!redTaskCApi.isActionSequenceRunning()) {
                redTaskCApi.triggerActionSequence();
            }*/

            /**用于仅抓取蓝色
            blueTaskCApi.update();
            if (!blueTaskCApi.isActionSequenceRunning()) {
                blueTaskCApi.triggerActionSequence(); // Call the API method
            }*/

            taskBApi.update();
            if (gamepad1.right_bumper && !taskBApi.isActionSequenceRunning()) {
                taskBApi.triggerActionSequence(); // Call the API method
            }
        }
        /**redTaskCApi.cleanup();*/
        /**blueTaskCApi.cleanup();*/
        taskBApi.cleanup();
    }
}