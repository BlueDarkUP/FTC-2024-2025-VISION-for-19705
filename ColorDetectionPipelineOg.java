// ColorDetectionPipeline.java
package org.firstinspires.ftc.teamcode.opmode.auto;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class ColorDetectionPipelineOg extends OpenCvPipeline {

    // Constants and Configuration
    private static final double PIXELS_TO_CM_RATIO = 0.1;
    private String allianceColor = "blue"; // Default to blue, can be set to "red" or "blue"
    private String taskCategory = "B"; // Default to Task B, can be set to "B" or "C"
    private static final double CENTER_RECTANGLE_WIDTH_CM = 24;
    private static final double CENTER_RECTANGLE_HEIGHT_CM = 47;
    private static final double CENTER_RECTANGLE_OFFSET_X_CM = 0;
    private static final double CENTER_RECTANGLE_OFFSET_Y_CM = 5;
    private static final Scalar PERSISTENT_LINE_COLOR = new Scalar(255, 255, 255);
    private static final int PERSISTENT_LINE_THICKNESS = 2;
    private static final Scalar INTERSECTION_POINT_COLOR = new Scalar(255, 0, 255);
    private static final int INTERSECTION_POINT_THICKNESS = 2;
    private static final Scalar CONNECTION_LINE_COLOR = new Scalar(255, 255, 0);
    private static final int CONNECTION_LINE_THICKNESS = 2;
    private static final int RED_LINE_THICKNESS = 2;
    private static final Scalar CENTER_TO_INTERSECTION_LINE_COLOR = new Scalar(255, 128, 0);
    private static final Scalar TEXT_COLOR = new Scalar(255, 255, 0);
    private static final Scalar PURPLE = new Scalar(255, 0, 255);
    private static final Scalar BLACK = new Scalar(0, 0, 0);

    private static final Scalar lower_red_hsv_1 = new Scalar(170, 100, 115);
    private static final Scalar upper_red_hsv_1 = new Scalar(180, 250, 255);
    private static final Scalar lower_red_hsv_2 = new Scalar(0, 100, 115);
    private static final Scalar upper_red_hsv_2 = new Scalar(10, 255, 255);
    private static final Scalar lower_blue_hsv = new Scalar(100, 125, 170);
    private static final Scalar upper_blue_hsv = new Scalar(130, 255, 255);
    private static final Scalar lower_yellow_hsv = new Scalar(20, 120, 170);
    private static final Scalar upper_yellow_hsv = new Scalar(40, 255, 255);

    private final Mat grey = new Mat(); // Reusable Mat for grayscale conversion
    private final Mat hsvImage = new Mat(); // Reusable Mat for HSV image
    private final Mat mask = new Mat(); // Reusable Mat for masks
    private final Mat hierarchy = new Mat(); // Reusable Mat for findContours hierarchy

    private final List<DetectedCube> detectedCubesList = new ArrayList<>(); // To hold detected cubes info
    private final Mat outputMat = new Mat(); // Reusable Mat for output image
    private DetectedCube closestCubeForTelemetry; // 用于存储被选中的最接近的立方体供外部访问


    public static class DetectedCube {
        String color;
        int centerXImagePx;
        int centerYImagePx;
        double centerXCm;
        double centerYCm;
        double angleDegrees;
        Rect boundingBox;
        double aspectRatio;
        double intersectionDistanceCm;
        double angleToVertical;

        public DetectedCube(String color, int centerXImagePx, int centerYImagePx, double centerXCm, double centerYCm, double angleDegrees, Rect boundingBox, double aspectRatio, double intersectionDistanceCm, double angleToVertical) {
            this.color = color;
            this.centerXImagePx = centerXImagePx;
            this.centerYImagePx = centerYImagePx;
            this.centerXCm = centerXCm;
            this.centerYCm = centerYCm;
            this.angleDegrees = angleDegrees;
            this.boundingBox = boundingBox;
            this.aspectRatio = aspectRatio;
            this.intersectionDistanceCm = intersectionDistanceCm;
            this.angleToVertical = angleToVertical;
        }

        // Getters (optional, add if you need to access cube properties)
        public String getColor() { return color; }
        public int getCenterXImagePx() { return centerXImagePx; }
        public int getCenterYImagePx() { return centerYImagePx; }
        public double getCenterXCm() { return centerXCm; }
        public double getCenterYCm() { return centerYCm; }
        public double getAngleDegrees() { return angleDegrees; }
        public Rect getBoundingBox() { return boundingBox; }
        public double getIntersectionDistanceCm() { return intersectionDistanceCm; }
        public double getAngleToVertical() { return angleToVertical; }
    }

    // Helper Functions
    private static double toCm(int pixels) {
        return pixels * PIXELS_TO_CM_RATIO;
    }

    private static int toPx(double cm) {
        return (int) (cm / PIXELS_TO_CM_RATIO);
    }

    private static double[] calculateWorldCoordinates(int centerXPx, int centerYPx, int imageHeight, double ratio) {
        double xCm = (centerXPx - imageHeight / 2.0) * ratio;
        double yCm = ((imageHeight - centerYPx) * ratio);
        return new double[]{xCm, yCm};
    }

    public void setAllianceColor(String color) {
        if (color.equalsIgnoreCase("red") || color.equalsIgnoreCase("blue")) {
            allianceColor = color.toLowerCase();
        } else {
            allianceColor = "blue"; // Default if invalid color provided
        }
    }

    public void setTaskCategory(String task) {
        if (task.equalsIgnoreCase("B") || task.equalsIgnoreCase("C")) {
            taskCategory = task.toUpperCase();
        } else {
            taskCategory = "B"; // Default if invalid task provided
        }
    }

    @Override
    public Mat processFrame(Mat input) {
        input.copyTo(outputMat); // Copy input to outputMat for drawing on, avoid modifying input directly
        detectedCubesList.clear(); // Clear the list from previous frame
        Imgproc.cvtColor(outputMat, hsvImage, Imgproc.COLOR_RGB2HSV);// EasyOpenCV gives RGBA, convert to HSV

        int height = outputMat.rows();
        int width = outputMat.cols();

        // Pre-calculate fixed values
        int centerXPxBase = width / 2;
        int centerYPxBase = height / 2;
        int offsetXPx = toPx(CENTER_RECTANGLE_OFFSET_X_CM);
        int offsetYPx = toPx(CENTER_RECTANGLE_OFFSET_Y_CM);
        int centerXPxRegion = centerXPxBase + offsetXPx;
        int centerYPxRegion = centerYPxBase + offsetYPx;
        int rectWidthPx = toPx(CENTER_RECTANGLE_WIDTH_CM);
        int rectHeightPx = toPx(CENTER_RECTANGLE_HEIGHT_CM);
        int semicircleRadiusPx = rectWidthPx / 2;
        int topLeftX = centerXPxRegion - rectWidthPx / 2;
        int topLeftY = centerYPxRegion - rectHeightPx / 2;
        int bottomRightX = centerXPxRegion + rectWidthPx / 2;
        int bottomRightY = centerYPxRegion + rectHeightPx / 2;
        int upperSemicircleCenterY = topLeftY;
        int lowerSemicircleCenterX = centerXPxRegion;
        int lowerSemicircleCenterY = bottomRightY;
        int lineX = width / 2 + offsetXPx; // Persistent line X position

        // Color Masks and Detection
        detectColor("red", new Scalar[]{lower_red_hsv_1, lower_red_hsv_2}, new Scalar[]{upper_red_hsv_1, upper_red_hsv_2}, lineX, centerXPxBase, centerYPxBase, centerXPxRegion, upperSemicircleCenterY, lowerSemicircleCenterX, lowerSemicircleCenterY, semicircleRadiusPx, topLeftX, topLeftY, bottomRightX, bottomRightY);
        detectColor("blue", new Scalar[]{lower_blue_hsv}, new Scalar[]{upper_blue_hsv}, lineX, centerXPxBase, centerYPxBase, centerXPxRegion, upperSemicircleCenterY, lowerSemicircleCenterX, lowerSemicircleCenterY, semicircleRadiusPx, topLeftX, topLeftY, bottomRightX, bottomRightY);
        detectColor("yellow", new Scalar[]{lower_yellow_hsv}, new Scalar[]{upper_yellow_hsv}, lineX, centerXPxBase, centerYPxBase, centerXPxRegion, upperSemicircleCenterY, lowerSemicircleCenterX, lowerSemicircleCenterY, semicircleRadiusPx, topLeftX, topLeftY, bottomRightX, bottomRightY);

        // Drawing persistent lines and center region markers
        Imgproc.line(outputMat, new Point(0, centerYPxRegion), new Point(width, centerYPxRegion), PERSISTENT_LINE_COLOR, PERSISTENT_LINE_THICKNESS);
        Imgproc.line(outputMat, new Point(centerXPxRegion, 0), new Point(centerXPxRegion, height), PERSISTENT_LINE_COLOR, PERSISTENT_LINE_THICKNESS);

        Point intersectionPointAbove = new Point(centerXPxRegion, centerYPxRegion - semicircleRadiusPx);
        Imgproc.circle(outputMat, intersectionPointAbove, 5, PURPLE, -1);

        Imgproc.ellipse(outputMat, new Point(centerXPxRegion, topLeftY), new Size(semicircleRadiusPx, semicircleRadiusPx), 0, 180, 360, BLACK, RED_LINE_THICKNESS);
        Imgproc.ellipse(outputMat, new Point(centerXPxRegion, bottomRightY), new Size(semicircleRadiusPx, semicircleRadiusPx), 0, 0, -180, BLACK, RED_LINE_THICKNESS);
        Imgproc.line(outputMat, new Point(topLeftX, topLeftY), new Point(topLeftX, bottomRightY), BLACK, RED_LINE_THICKNESS);
        Imgproc.line(outputMat, new Point(bottomRightX, topLeftY), new Point(bottomRightX, bottomRightY), BLACK, RED_LINE_THICKNESS);

        mask.setTo(new Scalar(0)); // Clear the mask
        MatOfPoint rectPoints = new MatOfPoint(new Point(topLeftX, topLeftY), new Point(bottomRightX, topLeftY), new Point(bottomRightX, bottomRightY), new Point(topLeftX, bottomRightY));
        Imgproc.fillPoly(mask, Collections.singletonList(rectPoints), new Scalar(255));
        Imgproc.ellipse(mask, new Point(centerXPxRegion, topLeftY), new Size(semicircleRadiusPx, semicircleRadiusPx), 0, 180, 360, new Scalar(255), -1);

        Mat tempOutputMat = new Mat(); // 临时 Mat 存储掩码结果
        Core.bitwise_and(outputMat, mask, tempOutputMat);
        tempOutputMat.copyTo(outputMat); // 将结果复制回 outputMat，用于后续显示
        tempOutputMat.release(); // 释放临时 Mat


        // Example of selecting and drawing closest cube
        DetectedCube closestCube = selectClosestCube(detectedCubesList, Arrays.asList(allianceColor), intersectionPointAbove, taskCategory);
        closestCubeForTelemetry = closestCube; // 将选中的立方体赋值给成员变量
        if (closestCube != null) {
            Imgproc.rectangle(outputMat, closestCube.getBoundingBox(), new Scalar(255, 0, 0), 2); // Red for closest (实际上是蓝色)
        }

        return outputMat;
    }

    private void detectColor(String colorName, Scalar[] lowerHsvRanges, Scalar[] upperHsvRanges, int lineX, int centerXPxBase, int centerYPxBase, int upperSemicircleCenterX, int upperSemicircleCenterY, int lowerSemicircleCenterX, int lowerSemicircleCenterY, int semicircleRadiusPx, int topLeftX, int topLeftY, int bottomRightX, int bottomRightY) {
        mask.setTo(new Scalar(0)); // Clear mask for each color

        try { // 添加 try-catch 块
            for (int i = 0; i < lowerHsvRanges.length; i++) {
                Core.inRange(hsvImage, lowerHsvRanges[i], upperHsvRanges[i], grey); // Use grey Mat as temporary mask
                Core.bitwise_or(mask, grey, mask); // Accumulate masks if multiple ranges for one color
            }

            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            for (MatOfPoint cnt : contours) {
                if (Imgproc.contourArea(cnt) > 500) {
                    RotatedRect rect = Imgproc.minAreaRect(new MatOfPoint2f(cnt.toArray()));
                    Size rectSize = rect.size;
                    double wRect = rectSize.width;
                    double hRect = rectSize.height;
                    Rect boundingBox = Imgproc.boundingRect(cnt);
                    int x = boundingBox.x;
                    int y = boundingBox.y;
                    int w = boundingBox.width;
                    int h = boundingBox.height;
                    int centerX = x + w / 2;
                    int centerY = y + h / 2;

                    double distSqLowerSemi = Math.pow(centerX - lowerSemicircleCenterX, 2) + Math.pow(centerY - lowerSemicircleCenterY, 2);
                    double distSqUpperSemi = Math.pow(centerX - upperSemicircleCenterX, 2) + Math.pow(centerY - upperSemicircleCenterY, 2);

                    boolean isInTransparentRedSemicircle = distSqLowerSemi <= Math.pow(semicircleRadiusPx, 2);
                    if (isInTransparentRedSemicircle) {
                        continue;
                    }

                    boolean isCenterValid = (topLeftX <= centerX && centerX <= bottomRightX && topLeftY <= centerY && centerY <= bottomRightY) ||
                            (centerY <= topLeftY && distSqUpperSemi <= Math.pow(semicircleRadiusPx, 2));

                    boolean isInLowerSemicircleInvalidRegion = centerY >= lowerSemicircleCenterY && distSqLowerSemi <= Math.pow(semicircleRadiusPx, 2);

                    if (!isCenterValid || isInLowerSemicircleInvalidRegion) {
                        continue;
                    }

                    double aspectRatioDetected = (wRect > 0 && hRect > 0) ? Math.min(wRect, hRect) / Math.max(wRect, hRect) : 0.0;
                    double angleDegrees = rect.angle;
                    if (wRect < hRect) {
                        angleDegrees += 90;
                    }
                    angleDegrees = angleDegrees % 180;


                    double intersectionPointDistanceCm = 0.0;
                    double angleToVertical = 0.0;
                    Point lowestIntersectionPoint = null;

                    double deltaSq = Math.pow(semicircleRadiusPx, 2) - Math.pow(lineX - centerX, 2); // Corrected: using semicircleRadiusPx
                    if (deltaSq >= 0) {
                        double delta = Math.sqrt(deltaSq);
                        int yIntersect1 = (int) (centerY + delta);
                        int yIntersect2 = (int) (centerY - delta);
                        List<Point> intersectionPoints = Arrays.asList(new Point(lineX, yIntersect1), new Point(lineX, yIntersect2));
                        lowestIntersectionPoint = Collections.max(intersectionPoints, (p1, p2) -> (int)(p1.y - p2.y)); // Lambda for Point comparison
                    }

                    if (lowestIntersectionPoint != null) {
                        Imgproc.circle(outputMat, lowestIntersectionPoint, 3, CONNECTION_LINE_COLOR, INTERSECTION_POINT_THICKNESS);
                        intersectionPointDistanceCm = toCm((int)(lowestIntersectionPoint.y - centerYPxBase - toPx(CENTER_RECTANGLE_OFFSET_Y_CM)));

                        double dx = lowestIntersectionPoint.x - centerX;
                        double dy = lowestIntersectionPoint.y - centerY;
                        angleToVertical = Math.toDegrees(Math.atan2(dx, dy));
                        if (dx < 0) angleToVertical = -Math.abs(angleToVertical);
                        else angleToVertical = Math.abs(angleToVertical);
                    }

                    if (lowestIntersectionPoint != null) {
                        Imgproc.line(outputMat, new Point(centerX, centerY), lowestIntersectionPoint, CENTER_TO_INTERSECTION_LINE_COLOR, CONNECTION_LINE_THICKNESS);
                    }

                    detectedCubesList.add(new DetectedCube(colorName, centerX, centerY, calculateWorldCoordinates(centerX, centerY, outputMat.rows(), PIXELS_TO_CM_RATIO)[0], calculateWorldCoordinates(centerX, centerY, outputMat.rows(), PIXELS_TO_CM_RATIO)[1], angleDegrees, boundingBox, aspectRatioDetected, intersectionPointDistanceCm, angleToVertical));

                    Imgproc.rectangle(outputMat, boundingBox, new Scalar(0, 255, 0), 2);
                    List<String> labels = Arrays.asList(colorName, String.format("%.1fdeg", angleDegrees), String.format("D:%.1fcm", intersectionPointDistanceCm), String.format("A:%.1fdeg", angleToVertical));
                    for (int i = 0; i < labels.size(); i++) {
                        Imgproc.putText(outputMat, labels.get(i), new Point(x, y - 10 * (i + 1)), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2);
                    }
                    Imgproc.circle(outputMat, new Point(centerX, centerY), 2, INTERSECTION_POINT_COLOR, -1);
                }
            }
        } catch (Exception e) {
            // 捕获 detectColor 方法内部的异常
            android.util.Log.e("ColorDetectionPipeline", "Error in detectColor for color: " + colorName, e);
            // 可以选择是否继续处理其他颜色或直接返回，这里选择继续
            // 如果需要更严格的错误处理，可以选择直接 return;
        } finally {
            // 在 finally 块中释放 Mat 对象 (虽然这里 grey, mask, hierarchy 是类成员变量，不需要在这里 release，
            // 但如果 detectColor 方法内部创建了临时的 Mat 对象，应该在这里 release)
            // 在当前代码中，detectColor 方法本身没有创建需要在此处释放的 Mat 对象。
        }
    }


    private DetectedCube selectClosestCube(List<DetectedCube> cubes, List<String> targetColors, Point referencePoint, String taskCategory) {
        DetectedCube closestCube = null;
        double minDistance = Double.MAX_VALUE;

        List<DetectedCube> filteredCubes = new ArrayList<>();

        if (taskCategory.equalsIgnoreCase("B")) {
            List<DetectedCube> yellowCubes = new ArrayList<>();
            for (DetectedCube cube : cubes) {
                if (cube.getColor().equalsIgnoreCase("yellow")) {
                    yellowCubes.add(cube);
                }
            }
            if (!yellowCubes.isEmpty()) {
                targetColors = Arrays.asList("yellow"); // Prioritize yellow for Task B
                filteredCubes = yellowCubes;
            } else {
                for (DetectedCube cube : cubes) {
                    if (targetColors.contains(cube.getColor())) {
                        filteredCubes.add(cube);
                    }
                }
            }
        } else if (taskCategory.equalsIgnoreCase("C")) {
            for (DetectedCube cube : cubes) {
                if (targetColors.contains(cube.getColor())) {
                    filteredCubes.add(cube);
                }
            }
        }


        if (filteredCubes.isEmpty()) {
            return null;
        }

        for (DetectedCube cube : filteredCubes) {
            double distance = Math.sqrt(Math.pow(cube.getCenterXImagePx() - referencePoint.x, 2) + Math.pow(cube.getCenterYImagePx() - referencePoint.y, 2));
            if (distance < minDistance) {
                minDistance = distance;
                closestCube = cube;
            }
        }
        return closestCube;
    }

    public List<DetectedCube> getDetectedCubes() {
        return detectedCubesList;
    }

    public DetectedCube getClosestCube() {
        return closestCubeForTelemetry;
    }
}