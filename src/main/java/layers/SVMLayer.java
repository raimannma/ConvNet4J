package layers;

import com.google.gson.JsonObject;
import utils.ParamsAndGrads;
import utils.Utils;
import utils.Vol;

public class SVMLayer extends Layer {
    private int numInputs;

    public SVMLayer() {
        this(new LayerConfig());
    }

    public SVMLayer(final LayerConfig opt) {
        // computed
        this.numInputs = opt.getInSX() * opt.getInSY() * opt.getInDepth();
        this.outDepth = this.numInputs;
        this.outSX = 1;
        this.outSY = 1;
        this.type = Layer.LayerType.SVM;
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.inAct = vol;
        this.outAct = vol;
        return vol;
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0];
    }

    @Override
    public JsonObject toJSON() {
        final JsonObject json = new JsonObject();
        json.addProperty("outDepth", this.outDepth);
        json.addProperty("outSX", this.outSX);
        json.addProperty("outSY", this.outSY);
        json.addProperty("type", this.type.toString());
        json.addProperty("numInputs", this.numInputs);
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.outDepth = json.get("outDepth").getAsInt();
        this.outSX = json.get("outSX").getAsInt();
        this.outSY = json.get("outSY").getAsInt();
        this.type = Layer.LayerType.valueOf(json.get("type").getAsString());
        this.numInputs = json.get("numInputs").getAsInt();
    }

    @Override
    public double backward(final Vol output) {
        final Vol x = this.inAct;
        x.dw = Utils.zerosDouble(x.w.length);

        final int y = (int) output.get(0, 0, 0);
        final double yScore = x.w[y];
        final double margin = 1;
        double loss = 0;
        for (int i = 0; i < this.outDepth; i++) {
            if (y == i) {
                continue;
            }
            final double yDiff = -yScore + x.w[i] + margin;
            if (yDiff > 0) {
                x.dw[i]++;
                x.dw[y]--;
                loss += yDiff;
            }
        }
        return loss;
    }
}
