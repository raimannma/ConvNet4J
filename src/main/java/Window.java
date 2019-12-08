import java.util.ArrayList;
import java.util.List;

class Window<T> {
    private final ArrayList<T> values;
    private final int windowSize;

    Window(final int windowSize) {
        this.values = new ArrayList<>();
        this.windowSize = windowSize;
    }

    void addAll(final T[] elems) {
        for (final T elem : elems) {
            this.add(elem);
        }
    }

    void add(final T elem) {
        if (this.values.size() >= this.windowSize) {
            this.values.remove(0);
        }
        this.values.add(elem);
    }

    void addAll(final List<T> elems) {
        for (final T elem : elems) {
            this.add(elem);
        }
    }

    T get(final int index) {
        return this.values.get(index);
    }

}
