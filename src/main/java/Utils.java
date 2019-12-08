import java.util.*;

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

    public static List<Integer> zerosIntList(final int size) {
        final Integer[] arr = new Integer[size];
        Arrays.fill(arr, 0);
        return Arrays.asList(arr);
    }

    public static int[] zerosInt(final int size) {
        final int[] arr = new int[size];
        Arrays.fill(arr, 0);
        return arr;
    }

    public static <T> boolean arrContains(final T[] arr, final T elem) {
        for (final T t : arr) {
            if (t.equals(elem)) {
                return true;
            }
        }
        return false;
    }

    // create random permutation of numbers, in range [0...n-1]
    public static int[] randPermutation(final int n) {
        int i = n;
        int j = 0;
        int temp;
        final int[] arr = new int[n];
        Arrays.setAll(arr, q -> q);
        while (i-- != 0) {
            j = random.nextInt(i + 1);
            temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
        return arr;
    }

    public static <T> T weightedSample(final T[] lst, final double[] probabilities) {
        final double p = randDouble(0, 1);
        double cumProb = 0;
        for (int i = 0; i < lst.length; i++) {
            cumProb += probabilities[i];
            if (p < cumProb) {
                return lst[i];
            }
        }
        return null;
    }

    public static double randDouble(final double min, final double max) {
        return random.nextDouble() * (max - min) + min;
    }

    public static <T> T getOption(final Map<String, T> options, final String fieldName, final T defaultValue) {
        return options.getOrDefault(fieldName, defaultValue);
    }

    public static <T> T[] getOptions(final Map<String, T> options, final String[] fieldNames, final T defaultValue) {
        final List<T> list = new ArrayList<>();
        for (final String fieldName : fieldNames) {
            list.add(options.getOrDefault(fieldName, defaultValue));
        }
        return (T[]) list.toArray();
    }

    public static double max(final double[] arr) {
        return Objects.requireNonNull(getMaxMin(arr))[1];
    }

    public static double[] getMaxMin(final double[] arr) {
        if (arr.length == 0) {
            return null;
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


