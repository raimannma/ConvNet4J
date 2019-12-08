package layers;

import com.google.gson.JsonObject;
import utils.ParamsAndGrads;
import utils.Utils;
import utils.Vol;

public class RegressionLayer extends Layer {
    private int numInputs;

    public RegressionLayer() {
        this(new LayerConfig());
    }

    public RegressionLayer(final LayerConfig opt) {
        //computed
        this.numInputs = opt.getInSX() * opt.getInSY() * opt.getInDepth();
        this.outDepth = this.numInputs;
        this.outSX = 1;
        this.outSY = 1;
        this.type = Layer.LayerType.REGRESSION;
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
        double loss = 0;
        if (output.sx == 1 && output.sy == 1 && output.depth == 1) {
            //single number
            final double y = output.get(0, 0, 0);
            final double dy = x.w[0] - y;
            x.dw[0] = dy;
            loss += 0.5 * dy * dy;
        } else if (output.sx == 1 && output.sy == 1 && output.depth == 2) {
            final int i = (int) output.get(0, 0, 0);
            final double y = output.get(0, 0, 1);
            final double dy = x.w[i] - y;
            x.dw[i] = dy;
            loss += 0.5 * dy * dy;
        } else if (output.sx == 1 && output.sy == 1) {
            for (int i = 0; i < this.outDepth; i++) {
                final double dy = x.w[i] - output.get(0, 0, i);
                x.dw[i] = dy;
                loss += 0.5 * dy * dy;
            }
        } else {
            throw new ArrayIndexOutOfBoundsException("Bad structure!");
        }
        return loss;
    }
}
