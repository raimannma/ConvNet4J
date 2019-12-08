import com.google.gson.JsonObject;

public class SoftmaxLayer extends Layer {
    private int numInputs;
    private double[] es;

    SoftmaxLayer() {
        this(new LayerConfig());
    }

    SoftmaxLayer(final LayerConfig opt) {
        //computed
        this.numInputs = opt.getInSX() * opt.getInSY() * opt.getInDepth();
        this.out_depth = this.numInputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.type = LayerType.SOFTMAX;
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.in_act = vol;

        final Vol outActivation = new Vol(1, 1, this.out_depth, 0);

        final double maxActivation = Utils.max(vol.w);

        final double[] es = new double[this.out_depth];
        double eSum = 0;
        for (int i = 0; i < es.length; i++) {
            final double e = Math.exp(vol.w[i] - maxActivation);
            eSum += e;
            es[i] = e;
        }
        for (int i = 0; i < es.length; i++) {
            es[i] /= eSum;
            outActivation.w[i] = es[i];
        }

        this.es = es;
        this.out_act = outActivation;
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
        json.addProperty("numInputs", this.numInputs);
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.out_depth = json.get("out_depth").getAsInt();
        this.out_sx = json.get("out_sx").getAsInt();
        this.out_sy = json.get("out_sy").getAsInt();
        this.type = LayerType.valueOf(json.get("type").getAsString());
        this.numInputs = json.get("numInputs").getAsInt();
    }

    @Override
    public double backward(final Vol output) {
        final Vol x = this.in_act;
        x.dw = Utils.zerosDouble(x.w.length);
        for (int i = 0; i < this.out_depth; i++) {
            final int indicator = i == (int) output.get(0, 0, 0) ? 1 : 0;
            x.dw[i] = this.es[i] - indicator;
        }
        return Math.log(this.es[(int) output.get(0, 0, 0)]);
    }
}
