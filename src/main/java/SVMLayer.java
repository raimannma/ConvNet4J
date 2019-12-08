import com.google.gson.JsonObject;

public class SVMLayer extends Layer {
    private int numInputs;

    SVMLayer() {
        this(new LayerConfig());
    }

    SVMLayer(final LayerConfig opt) {
        // computed
        this.numInputs = opt.getInSX() * opt.getInSY() * opt.getInDepth();
        this.out_depth = this.numInputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.type = LayerType.SVM;
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

        final int y = (int) output.get(0, 0, 0);
        final double yScore = x.w[y];
        final double margin = 1;
        double loss = 0;
        for (int i = 0; i < this.out_depth; i++) {
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
