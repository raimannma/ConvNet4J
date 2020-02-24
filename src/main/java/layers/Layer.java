package layers;

import com.google.gson.JsonObject;
import utils.ParamsAndGrads;
import utils.Vol;

public abstract class Layer {
    protected Vol outAct;
    protected Vol inAct;
    protected int outSX;
    protected int outSY;
    protected int outDepth;
    protected LayerType type;

    public Vol forward(final Vol vol) {
        return this.forward(vol, false);
    }

    public abstract Vol forward(Vol vol, boolean isTraining);

    public abstract ParamsAndGrads[] getParamsAndGrads();

    public abstract JsonObject toJSON();

    public abstract void fromJSON(JsonObject jsonLayer);

    public abstract double backward(Vol output);

    public Vol getOutAct() {
        return this.outAct;
    }

    public void setOutAct(final Vol outAct) {
        this.outAct = outAct;
    }

    public Vol getInAct() {
        return this.inAct;
    }

    public void setInAct(final Vol inAct) {
        this.inAct = inAct;
    }

    public int getOutSX() {
        return this.outSX;
    }

    public void setOutSX(final int outSX) {
        this.outSX = outSX;
    }

    public int getOutSY() {
        return this.outSY;
    }

    public void setOutSY(final int outSY) {
        this.outSY = outSY;
    }

    public int getOutDepth() {
        return this.outDepth;
    }

    public void setOutDepth(final int outDepth) {
        this.outDepth = outDepth;
    }

    public LayerType getType() {
        return this.type;
    }

    public void setType(final LayerType type) {
        this.type = type;
    }

    public enum LayerType {
        FC, LRN, DROPOUT, SOFTMAX, REGRESSION, CONVOLUTIONAL, POOL, RELU, SIGMOID, TANH, MAXOUT, INPUT, SVM
    }
}
