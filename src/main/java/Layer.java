abstract class Layer {
    public Vol out_act;
    LayerType type;

    public abstract Vol forward(Vol vol, boolean isTraining);

    public abstract ParamsAndGrads[] getParamsAndGrads();

    public enum LayerType {
        FC, LRN, DROPOUT, SOFTMAX, REGRESSION, CONVOLUTIONAL, POOL, RELU, SIGMOID, TANH, MAXOUT, INPUT, SVM
    }
}
