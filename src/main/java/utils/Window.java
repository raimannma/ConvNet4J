package utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Window<T> {
    private final int windowSize;
    public ArrayList<T> values;

    public Window(final int windowSize) {
        this.values = new ArrayList<>();
        this.windowSize = windowSize;
    }

    void addAll(final T[] elems) {
        Arrays.stream(elems).forEach(this::add);
    }

    public void add(final T elem) {
        if (this.values.size() >= this.windowSize) {
            this.values.remove(0);
            this.values = new ArrayList<>(this.values);
        }
        this.values.add(elem);
    }

    void addAll(final List<T> elems) {
        elems.forEach(this::add);
    }

    public T get(final int index) {
        return this.values.get(index);
    }
}
