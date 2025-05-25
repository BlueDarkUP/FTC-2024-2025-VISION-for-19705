package opmode.auto; // Or your actual package

import android.annotation.SuppressLint;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;
import com.qualcomm.robotcore.hardware.Servo;
import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvCamera;
import org.openftc.easyopencv.OpenCvCameraFactory;
import org.openftc.easyopencv.OpenCvCameraRotation;
import org.openftc.easyopencv.OpenCvPipeline;
import org.openftc.easyopencv.OpenCvWebcam; // Ensure this import is present

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

@TeleOp(name = "视觉识别程序") // Updated name to reflect change
public class WebcamExample2 extends LinearOpMode {
    private Servo inFlip;
    private Servo inFlipTurn;

    // --- Configuration Constants ---
    private static final String TASK_CATEGORY = "B";
    private static final String ALLIANCE_COLOR = "red";
    private static final String WEBCAM_NAME = "Webcam";
    private static final OpenCvCameraRotation CAMERA_STREAMING_ROTATION = OpenCvCameraRotation.UPSIDE_DOWN;
    private static final boolean IMAGE_ROTATE_180 = true;
    private static final double DOWNSCALE_FACTOR = 1.0;
    private static final double PIXELS_TO_CM_RATIO = 0.033984375;
    private static final double ARM_BACK_CM = 11.9;
    private static final double ERROR_CM = 7.0;
    private static final double CIRCLE_CENTER_OFFSET_CM_X = 0.0;
    private static final double CIRCLE_CENTER_OFFSET_CM_Y = 12.9;
    private static final double GRASPABLE_ANGLE_DEG = 170.0;
    private static final Map<String, List<Scalar[]>> HSV_COLOR_RANGES = new HashMap<>();
    static {
        HSV_COLOR_RANGES.put("red", List.of(
                new Scalar[]{new Scalar(170, 100, 120), new Scalar(180, 255, 255)},
                new Scalar[]{new Scalar(0, 100, 120), new Scalar(10, 255, 255)}
        ));
        HSV_COLOR_RANGES.put("blue", Collections.singletonList(
                new Scalar[]{new Scalar(100, 125, 160), new Scalar(130, 255, 255)}
        ));
        HSV_COLOR_RANGES.put("yellow", Collections.singletonList(
                new Scalar[]{new Scalar(20, 100, 170), new Scalar(40, 255, 255)}
        ));
    }
    private static final double MIN_SIZE_PIXELS = 500;
    private static final double MAX_SIZE_PIXELS = 100000;
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
    // --- End Configuration Constants ---

    OpenCvWebcam webcam;
    private ColorDetectionPipelineImpl colorDetectionPipeline;
    private final int cameraWidth = 1280;
    private final int cameraHeight = 720;
    private volatile boolean streamingSuccessfullyStarted = false; // Flag to track streaming status

    @SuppressLint({"DefaultLocale", "SetTextI18n"})
    @Override
    public void runOpMode() throws InterruptedException {
        printConfig();

        int cameraMonitorViewId = hardwareMap.appContext.getResources().getIdentifier(
                "cameraMonitorViewId", "id", hardwareMap.appContext.getPackageName());

        try {
            webcam = OpenCvCameraFactory.getInstance().createWebcam(
                    hardwareMap.get(WebcamName.class, WEBCAM_NAME), cameraMonitorViewId);
        } catch (IllegalArgumentException e) {
            telemetry.addLine("*** FATAL ERROR: Webcam '" + WEBCAM_NAME + "' not found in configuration! ***");
            telemetry.update();
            sleep(5000);
            requestOpModeStop();
            return;
        }

        colorDetectionPipeline = new ColorDetectionPipelineImpl();
        webcam.setPipeline(colorDetectionPipeline);
        webcam.setMillisecondsPermissionTimeout(5000);

        telemetry.addLine("Opening camera asynchronously...");
        telemetry.update();

        // Ensure flag starts false
        streamingSuccessfullyStarted = false;

        webcam.openCameraDeviceAsync(new OpenCvCamera.AsyncCameraOpenListener() {
            @Override
            public void onOpened() {
                telemetry.addLine("Camera opened successfully.");
                telemetry.addLine(String.format("Attempting to start stream: %dx%d @ %s using MJPEG format",
                        cameraWidth, cameraHeight, CAMERA_STREAMING_ROTATION));
                telemetry.update();

                try {
                    webcam.startStreaming(
                            cameraWidth,
                            cameraHeight,
                            CAMERA_STREAMING_ROTATION,
                            OpenCvWebcam.StreamFormat.MJPEG
                    );
                    streamingSuccessfullyStarted = true; // Set flag on success
                    telemetry.addLine("Streaming started successfully (using MJPEG).");
                    telemetry.update();

                } catch (Exception e) {
                    streamingSuccessfullyStarted = false; // Ensure flag is false on failure
                    telemetry.addData("ERROR", "Could not start MJPEG streaming.");
                    telemetry.addData("Reason", "Camera might not support MJPEG at " + cameraWidth + "x" + cameraHeight + ", or another error occurred.");
                    telemetry.addData("Exception", e.getMessage());
                    telemetry.update();
                }
            }

            @Override
            public void onError(int errorCode) {
                streamingSuccessfullyStarted = false; // Ensure flag is false on open error
                telemetry.addData("FATAL ERROR", "Camera could not be opened. Error Code: " + errorCode);
                telemetry.update();
            }
        });

        inFlip = hardwareMap.get(Servo.class, "inFlip");
        inFlipTurn = hardwareMap.get(Servo.class, "inFlipTurn");
        inFlip.setPosition(0.68);
        inFlipTurn.setPosition(0.5);

        telemetry.addLine("Waiting for start (camera init running in background)...");
        telemetry.addData("Task Mode", TASK_CATEGORY);
        telemetry.addData("Alliance Color", ALLIANCE_COLOR);
        telemetry.update();

        waitForStart();

        while (opModeIsActive()) {
            // --- Telemetry Section ---
            telemetry.addLine(opModeIsActive() ? "--- OpMode Running ---" : "--- OpMode Ending ---");

            if (webcam != null) {
                if (streamingSuccessfullyStarted) { // Use the flag
                    try {
                        telemetry.addData("FPS", String.format(Locale.US, "%.2f", webcam.getFps()));
                        telemetry.addData("Pipeline Time (ms)", webcam.getPipelineTimeMs());
                        telemetry.addData("Requested Resolution", cameraWidth + "x" + cameraHeight); // Show requested size
                    } catch (Exception e) {
                        telemetry.addData("Webcam Info","Error getting stats: " + e.getMessage());
                    }
                } else {
                    telemetry.addData("Webcam Status", "Opened, but streaming did NOT start successfully (check logs)");
                }
            } else {
                telemetry.addData("Webcam Status", "Not Initialized or Failed to Open");
            }

            // --- Detection Results from Pipeline (No changes needed here) ---
            telemetry.addLine("--- Detection & Selection ---");
            if (colorDetectionPipeline != null) {
                List<DetectedCube> allCubes = colorDetectionPipeline.getAllDetectedCubes();
                DetectedCube selectedCube = colorDetectionPipeline.getSelectedCube();
                int detectedCount = allCubes.size();
                telemetry.addData("Detected Cubes (All Colors)", detectedCount);

                if (selectedCube != null) {
                    telemetry.addLine("--- Final Target (Red Box) ---");
                    telemetry.addData("  Color", selectedCube.color);
                    telemetry.addData("  Image Coords (px)", String.format("(%d, %d)", selectedCube.centerXImagePx, selectedCube.centerYImagePx));
                    telemetry.addData("  Angle from Vertical (deg)", String.format(Locale.US, "%.1f", selectedCube.angleFromVertical));
                    telemetry.addData("  In Ring?", selectedCube.isInRing ? "Yes" : "No");
                    telemetry.addData("  In Sector?", selectedCube.isInGraspAngle ? "Yes" : "No");
                    telemetry.addData("  Area (px^2)", String.format(Locale.US, "%.0f", selectedCube.areaPx));
                } else {
                    telemetry.addLine("--- No Suitable Target Selected ---");
                    if (!allCubes.isEmpty()) {
                        int potentialCandidateCount = 0;
                        for (DetectedCube cube : allCubes) {
                            boolean isPotential = TASK_CATEGORY.equals("B")
                                    ? cube.isInRing && cube.isInGraspAngle && (cube.color.equals("yellow") || cube.color.equals(ALLIANCE_COLOR))
                                    : cube.isInRing && cube.color.equals(ALLIANCE_COLOR);
                            if (isPotential) potentialCandidateCount++;
                        }
                        telemetry.addData("  Potential Candidates Found", potentialCandidateCount);
                        if (potentialCandidateCount == 0 && detectedCount > 0) {
                            telemetry.addLine("  Reason: Detected cubes didn't meet criteria.");
                        }
                    } else {
                        telemetry.addLine("  Reason: No objects detected.");
                    }
                }
                if (colorDetectionPipeline.isViewportPaused()) {
                    telemetry.addLine("*** Viewport Paused (Tap preview to resume) ***");
                }
            } else {
                telemetry.addLine("Pipeline not initialized.");
            }
            // --- End Detection Telemetry ---

            telemetry.update();
            // --- End Telemetry Section ---

            // --- Optional: Stop Streaming Example ---
            if (gamepad1.a && webcam != null && streamingSuccessfullyStarted) { // Use the flag
                telemetry.addLine("Gamepad A pressed: Stopping camera stream.");
                telemetry.update();
                webcam.stopStreaming();
                streamingSuccessfullyStarted = false; // Update flag after stopping
                sleep(1000);
            }

            if (!isStopRequested()) {
                sleep(20);
            }
        }

        // Cleanup handled by SDK mostly
        telemetry.addLine("OpMode stopped.");
        telemetry.update();
    }

    // --- printConfig() method (No changes needed) ---
    @SuppressLint("DefaultLocale")
    private void printConfig() {
        telemetry.addLine("===================== Vision Program Start =====================");
        telemetry.addLine("--- Configuration ---");
        telemetry.addData("Task Mode", TASK_CATEGORY);
        telemetry.addData("Alliance Color", ALLIANCE_COLOR);
        telemetry.addData("Webcam Name", WEBCAM_NAME);
        telemetry.addData("Requested Format", "MJPEG"); // Indicate requested format
        telemetry.addData("Resolution", String.format("%d x %d", cameraWidth, cameraHeight));
        telemetry.addData("Stream Rotation", CAMERA_STREAMING_ROTATION);
        telemetry.addData("Pipeline 180 Rotate", IMAGE_ROTATE_180);
        telemetry.addData("Downscale Factor", String.format(Locale.US, "%.2f", DOWNSCALE_FACTOR));
        telemetry.addData("Ring Inner Radius (cm)", String.format(Locale.US, "%.1f", ARM_BACK_CM));
        telemetry.addData("Ring Width (cm)", String.format(Locale.US, "%.1f", ERROR_CM));
        telemetry.addData("Center Offset X (cm)", String.format(Locale.US, "%.1f", CIRCLE_CENTER_OFFSET_CM_X));
        telemetry.addData("Center Offset Y (cm)", String.format(Locale.US, "%.1f", CIRCLE_CENTER_OFFSET_CM_Y));
        telemetry.addData("Graspable Angle (deg)", String.format(Locale.US, "%.1f", GRASPABLE_ANGLE_DEG));
        telemetry.addData("Pixels/cm Ratio (est)", String.format(Locale.US, "%.3f", PIXELS_TO_CM_RATIO));
        telemetry.addData("Min Area (px^2)", String.format(Locale.US, "%.0f", MIN_SIZE_PIXELS));
        telemetry.addData("Max Area (px^2)", String.format(Locale.US, "%.0f", MAX_SIZE_PIXELS));
        telemetry.addLine("--------------------------------------------------");
        telemetry.update();
    }

    // --- Utility Methods (No changes needed) ---
    private static double toCm(double pixels) { return pixels * PIXELS_TO_CM_RATIO; }
    private static int toPx(double cm) {
        if (cm <= 0 || PIXELS_TO_CM_RATIO <= 1e-9) return 0;
        int px = (int) Math.round(cm / PIXELS_TO_CM_RATIO);
        return Math.max(1, px);
    }
    private static double calculateAngleFromVertical(Point pointPx, Point centerPx) {
        double dx = pointPx.x - centerPx.x;
        double dy = centerPx.y - pointPx.y;
        double angleRad = Math.atan2(dx, dy);
        return Math.toDegrees(angleRad);
    }

    // --- Pipeline Implementation (No changes needed) ---
    class ColorDetectionPipelineImpl extends OpenCvPipeline {
        private final Mat rgbImage = new Mat();
        private final Mat hsvImage = new Mat();
        private final Map<String, Mat> individualMasks = new HashMap<>();
        private final Mat rangeMask = new Mat();
        private final Mat hierarchy = new Mat();
        private final List<DetectedCube> allDetectedCubes = new ArrayList<>();
        private DetectedCube selectedCube = null;
        public final Object processingLock = new Object();
        private volatile boolean viewportPaused = false;

        public DetectedCube getSelectedCube() {
            synchronized (processingLock) { return (selectedCube == null) ? null : new DetectedCube(selectedCube); }
        }
        public List<DetectedCube> getAllDetectedCubes() {
            synchronized (processingLock) {
                List<DetectedCube> copy = new ArrayList<>(allDetectedCubes.size());
                for (DetectedCube cube : allDetectedCubes) { copy.add(new DetectedCube(cube)); }
                return copy;
            }
        }
        public boolean isViewportPaused() { return viewportPaused; }

        @Override
        public Mat processFrame(Mat input) {
            Mat displayOutput = input;
            if (IMAGE_ROTATE_180) { Core.rotate(input, input, Core.ROTATE_180); }
            if (DOWNSCALE_FACTOR < 1.0 && DOWNSCALE_FACTOR > 0) {
                Imgproc.resize(input, input, new Size(), DOWNSCALE_FACTOR, DOWNSCALE_FACTOR, Imgproc.INTER_LINEAR);
                displayOutput = input;
            }
            int height = displayOutput.rows(); int width = displayOutput.cols();
            if (height <= 0 || width <= 0) { return input; }

            int armBackPx = toPx(ARM_BACK_CM); int errorPx = toPx(ERROR_CM); int armForwardPx = armBackPx + errorPx;
            int circleCenterOffsetPxX = toPx(CIRCLE_CENTER_OFFSET_CM_X); int circleCenterOffsetPxY = toPx(CIRCLE_CENTER_OFFSET_CM_Y);
            Point operationalCenterPx = new Point((double)width / 2.0 + circleCenterOffsetPxX, (double)height / 2.0 + circleCenterOffsetPxY);
            double halfGraspAngle = (0 < GRASPABLE_ANGLE_DEG && GRASPABLE_ANGLE_DEG < 360) ? GRASPABLE_ANGLE_DEG / 2.0 : 181.0;

            Imgproc.cvtColor(displayOutput, rgbImage, Imgproc.COLOR_RGBA2RGB); Imgproc.cvtColor(rgbImage, hsvImage, Imgproc.COLOR_RGB2HSV);
            List<DetectedCube> currentFrameCubes = new ArrayList<>();

            for (Map.Entry<String, List<Scalar[]>> entry : HSV_COLOR_RANGES.entrySet()) {
                String colorName = entry.getKey(); List<Scalar[]> ranges = entry.getValue();
                Mat currentColorMask = individualMasks.get(colorName);
                if (currentColorMask == null || currentColorMask.rows() != hsvImage.rows() || currentColorMask.cols() != hsvImage.cols()) {
                    if (currentColorMask != null) currentColorMask.release();
                    currentColorMask = new Mat(hsvImage.size(), CvType.CV_8UC1); individualMasks.put(colorName, currentColorMask);
                }
                currentColorMask.setTo(Scalar.all(0));
                for (Scalar[] range : ranges) {
                    if (range != null && range.length == 2 && range[0] != null && range[1] != null) {
                        Core.inRange(hsvImage, range[0], range[1], rangeMask); Core.bitwise_or(currentColorMask, rangeMask, currentColorMask);
                    }
                }
                List<MatOfPoint> contours = new ArrayList<>();
                Imgproc.findContours(currentColorMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                for (MatOfPoint contour : contours) {
                    double area = Imgproc.contourArea(contour);
                    if (area >= MIN_SIZE_PIXELS && area <= MAX_SIZE_PIXELS) {
                        Rect boundingRect = Imgproc.boundingRect(contour); Point centerPx = new Point(boundingRect.x + boundingRect.width / 2.0, boundingRect.y + boundingRect.height / 2.0);
                        double objectOrientationDeg = 0.0; double dx = centerPx.x - operationalCenterPx.x; double dy = centerPx.y - operationalCenterPx.y;
                        double distSq = dx * dx + dy * dy; boolean isInRing = (distSq >= (double)armBackPx * armBackPx && distSq <= (double)armForwardPx * armForwardPx);
                        double angleFromVertical = calculateAngleFromVertical(centerPx, operationalCenterPx); boolean isInGraspAngle = Math.abs(angleFromVertical) <= halfGraspAngle;
                        currentFrameCubes.add(new DetectedCube(colorName, (int) Math.round(centerPx.x), (int) Math.round(centerPx.y), objectOrientationDeg, boundingRect, area, isInRing, angleFromVertical, isInGraspAngle));
                    }
                    contour.release();
                }
            }

            DetectedCube currentSelectedCube = null; List<DetectedCube> candidates = new ArrayList<>();
            for (DetectedCube cube : currentFrameCubes) {
                boolean isCandidate;
                if (TASK_CATEGORY.equals("B")) { isCandidate = cube.isInRing && cube.isInGraspAngle && (cube.color.equals("yellow") || cube.color.equals(ALLIANCE_COLOR)); }
                else { isCandidate = cube.isInRing && cube.color.equals(ALLIANCE_COLOR); }
                if (isCandidate) { candidates.add(cube); }
            }
            if (!candidates.isEmpty()) {
                if (TASK_CATEGORY.equals("B")) { candidates.sort(Comparator.<DetectedCube, Integer>comparing(c -> c.color.equals("yellow") ? 0 : 1).thenComparingDouble(c -> Math.abs(c.angleFromVertical))); }
                else { candidates.sort(Comparator.comparingDouble(c -> Math.abs(c.angleFromVertical))); }
                currentSelectedCube = candidates.get(0);
            }
            synchronized (processingLock) { allDetectedCubes.clear(); allDetectedCubes.addAll(currentFrameCubes); selectedCube = currentSelectedCube; }

            drawStaticOverlays(displayOutput, operationalCenterPx, armBackPx, armForwardPx, halfGraspAngle);
            for (DetectedCube cube : currentFrameCubes) {
                Scalar boxColor = (cube == currentSelectedCube) ? BOX_COLOR_SELECTED : BOX_COLOR_DEFAULT;
                Imgproc.rectangle(displayOutput, cube.boundingBox, boxColor, 2); Imgproc.circle(displayOutput, new Point(cube.centerXImagePx, cube.centerYImagePx), 4, BLOCK_CENTER_CIRCLE_COLOR, -1);
                String labelColor = cube.color; String labelAngle = String.format(Locale.US, "V:%.1f", cube.angleFromVertical);
                Point textOrigin = new Point(cube.boundingBox.x, cube.boundingBox.y - 15); if (textOrigin.y < 25) { textOrigin.y = cube.boundingBox.y + cube.boundingBox.height + 25; }
                Imgproc.putText(displayOutput, labelColor, new Point(textOrigin.x, textOrigin.y - 12), Imgproc.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, Imgproc.LINE_AA);
                Imgproc.putText(displayOutput, labelAngle, textOrigin, Imgproc.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, Imgproc.LINE_AA);
            }
            if (currentSelectedCube != null) {
                Point selectedCenterPx = new Point(currentSelectedCube.centerXImagePx, currentSelectedCube.centerYImagePx);
                Imgproc.line(displayOutput, operationalCenterPx, selectedCenterPx, CENTER_TO_TARGET_LINE_COLOR, CONNECTION_LINE_THICKNESS, Imgproc.LINE_AA);
            }
            return displayOutput;
        }

        private void drawStaticOverlays(Mat image, Point center, int radiusBack, int radiusFwd, double halfAngleDeg) {
            int height = image.rows(); int centerX = (int) Math.round(center.x);
            Imgproc.line(image, new Point(centerX, 0), new Point(centerX, height), PERSISTENT_LINE_COLOR, PERSISTENT_LINE_THICKNESS);
            boolean drawArcs = (halfAngleDeg > 0 && halfAngleDeg <= 180.0 && radiusBack > 0 && radiusFwd > 0 && radiusFwd > radiusBack);
            if (drawArcs) {
                double startAngleCv = 270.0 - halfAngleDeg; double endAngleCv = 270.0 + halfAngleDeg;
                Size axesBack = new Size(radiusBack, radiusBack); Size axesFwd = new Size(radiusFwd, radiusFwd);
                Imgproc.ellipse(image, center, axesBack, 0, startAngleCv, endAngleCv, CIRCLE_COLOR, CIRCLE_THICKNESS, Imgproc.LINE_AA);
                Imgproc.ellipse(image, center, axesFwd, 0, startAngleCv, endAngleCv, CIRCLE_COLOR, CIRCLE_THICKNESS, Imgproc.LINE_AA);
                double angleRadLeft = Math.toRadians(270 - halfAngleDeg); double angleRadRight = Math.toRadians(270 + halfAngleDeg);
                Point endLeft = new Point(center.x + radiusFwd * Math.cos(angleRadLeft), center.y + radiusFwd * Math.sin(angleRadLeft));
                Point endRight = new Point(center.x + radiusFwd * Math.cos(angleRadRight), center.y + radiusFwd * Math.sin(angleRadRight));
                Imgproc.line(image, center, endLeft, GRASPABLE_ANGLE_LINE_COLOR, GRASPABLE_ANGLE_LINE_THICKNESS, Imgproc.LINE_AA);
                Imgproc.line(image, center, endRight, GRASPABLE_ANGLE_LINE_COLOR, GRASPABLE_ANGLE_LINE_THICKNESS, Imgproc.LINE_AA);
            } else {
                if (radiusBack > 0) { Imgproc.circle(image, center, radiusBack, CIRCLE_COLOR, CIRCLE_THICKNESS, Imgproc.LINE_AA); }
                if (radiusFwd > 0) { Imgproc.circle(image, center, radiusFwd, CIRCLE_COLOR, CIRCLE_THICKNESS, Imgproc.LINE_AA); }
            }
        }

        @Override
        public void onViewportTapped() {
            viewportPaused = !viewportPaused;
            if (webcam != null) { if (viewportPaused) { webcam.pauseViewport(); } else { webcam.resumeViewport(); } }
        }
    }
    // --- End Pipeline ---

    // --- DetectedCube Data Structure (No changes needed) ---
    public static class DetectedCube {
        final String color; final int centerXImagePx; final int centerYImagePx; final double objectOrientationDeg;
        final Rect boundingBox; final double areaPx; final boolean isInRing; final double angleFromVertical;
        final boolean isInGraspAngle;
        public DetectedCube(String color, int xPx, int yPx, double objAngle, Rect rect, double area, boolean inRing, double angleVert, boolean inAngle) {
            this.color = color; this.centerXImagePx = xPx; this.centerYImagePx = yPx; this.objectOrientationDeg = objAngle;
            this.boundingBox = rect.clone(); this.areaPx = area; this.isInRing = inRing; this.angleFromVertical = angleVert; this.isInGraspAngle = inAngle;
        }
        public DetectedCube(DetectedCube other) { // Copy constructor
            this.color = other.color; this.centerXImagePx = other.centerXImagePx; this.centerYImagePx = other.centerYImagePx; this.objectOrientationDeg = other.objectOrientationDeg;
            this.boundingBox = other.boundingBox.clone(); this.areaPx = other.areaPx; this.isInRing = other.isInRing; this.angleFromVertical = other.angleFromVertical; this.isInGraspAngle = other.isInGraspAngle;
        }
    }
    // --- End DetectedCube ---
}