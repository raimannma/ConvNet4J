import com.google.gson.JsonObject;

public class ReluLayer extends Layer {
    ReluLayer() {
        this(new LayerConfig());
    }

    ReluLayer(final LayerConfig opt) {
        //computed
        this.out_sx = opt.getInSX();
        this.out_sy = opt.getInSY();
        this.out_depth = opt.getInDepth();
        this.type = LayerType.RELU;
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.in_act = vol;
        final Vol vol2 = vol.clone();
        for (int i = 0; i < vol.w.length; i++) {
            if (vol2.w[i] < 0) {
                vol2.w[i] = 0;
            }
        }
        this.out_act = vol2;
        return this.out_act;
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
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.out_depth = json.get("out_depth").getAsInt();
        this.out_sx = json.get("out_sx").getAsInt();
        this.out_sy = json.get("out_sy").getAsInt();
        this.type = LayerType.valueOf(json.get("layer_type").getAsString());
    }

    @Override
    public double backward(final Vol output) {
        final Vol vol = this.in_act;
        final Vol vol2 = this.out_act;
        vol.dw = Utils.zerosDouble(vol.w.length);
        for (int i = 0; i < vol.w.length; i++) {
            vol.dw[i] = vol2.w[i] <= 0 ? 0 : vol2.dw[i];
        }
        return 0;
    }
}
