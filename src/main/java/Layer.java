import com.google.gson.JsonObject;

abstract class Layer {
    public Vol outAct;
    public Vol inAct;
    public int outSX;
    public int outSY;
    public int outDepth;
    LayerType type;

    public Vol forward(final Vol vol) {
        return this.forward(vol, false);
    }

    public abstract Vol forward(Vol vol, boolean isTraining);

    public abstract ParamsAndGrads[] getParamsAndGrads();

    public abstract JsonObject toJSON();

    public abstract void fromJSON(JsonObject jsonLayer);

    public abstract double backward(Vol output);

    public enum LayerType {
        FC, LRN, DROPOUT, SOFTMAX, REGRESSION, CONVOLUTIONAL, POOL, RELU, SIGMOID, TANH, MAXOUT, INPUT, SVM
    }
}
