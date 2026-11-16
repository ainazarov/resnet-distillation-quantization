package org.fooddiary.benchmark;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ColorSpace;
import android.os.Build;
import android.os.SystemClock;

import org.pytorch.executorch.Tensor;

import java.io.File;
import java.io.IOException;

/** The 255-short-side Pillow bilinear resize, 224 center crop and ImageNet normalization. */
final class Preprocess {
    private static final int SHORT_SIDE = 255;
    private static final int CROP = 224;
    private static final int PLANE = CROP * CROP;
    private static final int WEIGHT_BITS = 22;
    private static final int HALF = 1 << (WEIGHT_BITS - 1);
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};
    private static final float[][] NORMALIZED = new float[3][256];

    static {
        for (int channel = 0; channel < 3; channel++) {
            for (int pixel = 0; pixel < 256; pixel++) {
                NORMALIZED[channel][pixel] = (pixel / 255f - MEAN[channel]) / STD[channel];
            }
        }
    }

    private Preprocess() {}

    static Tensor fromJpeg(File jpeg, long[] decodedAt) throws IOException {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bitmap = BitmapFactory.decodeFile(jpeg.getPath(), options);
        if (bitmap == null) {
            throw new IOException("Could not decode JPEG " + jpeg);
        }
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        try {
            if (Build.VERSION.SDK_INT >= 29) {
                // Pillow convert("RGB") retains the RGB values of these JPEGs, including ICC-tagged images.
                bitmap.setColorSpace(ColorSpace.get(ColorSpace.Named.SRGB));
            }
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        } finally {
            bitmap.recycle();
        }
        if (decodedAt != null) decodedAt[0] = SystemClock.elapsedRealtimeNanos();
        return Tensor.fromBlob(normalizeCrop(pixels, width, height), new long[]{1, 3, CROP, CROP});
    }

    // Calculate only the central crop of the resized image. Pillow rounds to 8 bits after each pass.
    static float[] normalizeCrop(int[] pixels, int width, int height) {
        int shortSide = Math.min(width, height);
        Axis xAxis = new Axis(width, (int) ((long) width * SHORT_SIDE / shortSide));
        Axis yAxis = new Axis(height, (int) ((long) height * SHORT_SIDE / shortSide));
        int[] horizontal = new int[height * CROP];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < CROP; x++) {
                int start = y * width + xAxis.first[x];
                horizontal[y * CROP + x] = filter(pixels, start, 1, xAxis.weights[x]);
            }
        }
        float[] input = new float[3 * PLANE];
        for (int y = 0; y < CROP; y++) {
            for (int x = 0; x < CROP; x++) {
                int start = yAxis.first[y] * CROP + x;
                int pixel = filter(horizontal, start, CROP, yAxis.weights[y]);
                int offset = y * CROP + x;
                input[offset] = NORMALIZED[0][pixel >>> 16];
                input[offset + PLANE] = NORMALIZED[1][(pixel >>> 8) & 255];
                input[offset + 2 * PLANE] = NORMALIZED[2][pixel & 255];
            }
        }
        return input;
    }

    private static int filter(int[] pixels, int index, int stride, int[] weights) {
        int red = HALF;
        int green = HALF;
        int blue = HALF;
        for (int weight : weights) {
            int pixel = pixels[index];
            red += weight * ((pixel >>> 16) & 255);
            green += weight * ((pixel >>> 8) & 255);
            blue += weight * (pixel & 255);
            index += stride;
        }
        return ((red >>> WEIGHT_BITS) << 16) | ((green >>> WEIGHT_BITS) << 8) | (blue >>> WEIGHT_BITS);
    }

    private static final class Axis {
        final int[] first = new int[CROP];
        final int[][] weights = new int[CROP][];

        Axis(int source, int target) {
            double scale = source / (double) target;
            double radius = Math.max(1.0, scale);
            int cropOffset = (int) Math.rint((target - CROP) / 2.0);
            for (int i = 0; i < CROP; i++) {
                double center = (i + cropOffset + 0.5) * scale;
                int lo = Math.max(0, (int) (center - radius + 0.5));
                int hi = Math.min(source, (int) (center + radius + 0.5));
                first[i] = lo;
                double[] coefficients = new double[hi - lo];
                double total = 0.0;
                for (int j = lo; j < hi; j++) {
                    double weight = Math.max(0.0, 1.0 - Math.abs(j + 0.5 - center) / radius);
                    coefficients[j - lo] = weight;
                    total += weight;
                }
                weights[i] = new int[coefficients.length];
                for (int j = 0; j < coefficients.length; j++) {
                    weights[i][j] = (int) Math.round((coefficients[j] / total) * (1 << WEIGHT_BITS));
                }
            }
        }
    }
}
