import com.google.gson.JsonObject;

public class RegressionLayer extends Layer {
    private int numInputs;

    RegressionLayer() {
        this(new LayerConfig());
    }

    RegressionLayer(final LayerConfig opt) {
        //computed
        this.numInputs = opt.getInSX() * opt.getInSY() * opt.getInDepth();
        this.out_depth = this.numInputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.type = LayerType.REGRESSION;
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.in_act = vol;
        this.out_act = vol;
        return vol;
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0];
    }

    @Override
    public JsonObject toJSON() {
        final JsonObject json = new JsonObject();
        json.addProperty("out_depth", this.out_depth);
        json.addProperty("out_sx", this.out_sx);
        json.addProperty("out_sy", this.out_sy);
        json.addProperty("type", this.type.toString());
        json.addProperty("numInputs", this.numInputs);
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.out_depth = json.get("out_depth").getAsInt();
        this.out_sx = json.get("out_sx").getAsInt();
        this.out_sy = json.get("out_sy").getAsInt();
        this.type = LayerType.valueOf(json.get("layer_type").getAsString());
        this.numInputs = json.get("numInputs").getAsInt();
    }

    @Override
    public double backward(final Vol output) {
        final Vol x = this.in_act;
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
            for (int i = 0; i < this.out_depth; i++) {
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
