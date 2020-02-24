package utils;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public enum Utils {
    ;
    private static final Random random = new Random();

    public static int randInt(final int min, final int max) {
        return random.nextInt(max - min) + min;
    }

    public static double randNormal(final double mean, final double standardDeviation) {
        return mean + gaussRandom() * standardDeviation;
    }

    public static double gaussRandom() {
        return random.nextGaussian();
    }

    public static double[] zerosDouble(final int size) {
        final double[] arr = new double[size];
        Arrays.fill(arr, 0);
        return arr;
    }

    public static List<Double> zerosDoubleList(final int size) {
        final Double[] arr = new Double[size];
        Arrays.fill(arr, 0.0);
        return Arrays.asList(arr);
    }

    public static int[] zerosInt(final int size) {
        final int[] arr = new int[size];
        Arrays.fill(arr, 0);
        return arr;
    }

    public static double[] getMaxMin(final double[] arr) {
        if (arr.length == 0) {
            return new double[]{-1, Double.MIN_VALUE, -1, Double.MAX_VALUE};
        }
        double max = arr[0];
        double min = arr[0];
        double maxIndex = 0;
        double minIndex = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                maxIndex = i;
            }
            if (arr[i] < min) {
                min = arr[i];
                minIndex = i;
            }
        }
        return new double[]{maxIndex, max, minIndex, min};
    }

    public static double tanh(final double val) {
        final double y = Math.exp(2 * val);
        return (y - 1) / (y + 1);
    }
}


