import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.util.ArrayList;
import java.util.List;

public class FullyConnectedLayer extends Layer {
    private Vol biases;
    private ArrayList<Vol> filters;
    private double l1_decay_mul;
    private double l2_decay_mul;
    private int numInputs;
    private Vol in_act;

    FullyConnectedLayer() {
        this(new LayerConfig());
    }

    FullyConnectedLayer(final LayerConfig opt) {
        // required
        this.out_depth = LayerConfig.getOrDefault(opt.getNumNeurons(), opt.getFilters());


        // optional
        this.l1_decay_mul = LayerConfig.getOrDefault(0, opt.getL1DecayMul());
        this.l2_decay_mul = LayerConfig.getOrDefault(1, opt.getL2DecayMul());

        //computed
        this.numInputs = opt.getInSX() * opt.getInSY() * opt.getInDepth();
        this.out_sx = 1;
        this.out_sy = 1;
        this.type = LayerType.FC;

        //init
        final double bias = LayerConfig.getOrDefault(0, opt.getBiasPref());
        this.filters = new ArrayList<>();
        for (int i = 0; i < this.out_depth; i++) {
            this.filters.add(new Vol(1, 1, this.numInputs));
        }
        this.biases = new Vol(1, 1, this.out_depth, bias);
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.in_act = vol;
        final Vol A = new Vol(1, 1, this.out_depth, 0.0);
        final double[] Vw = vol.w;
        for (int i = 0; i < this.out_depth; i++) {
            double a = 0;
            final double[] wi = this.filters.get(i).w;
            for (int j = 0; j < this.numInputs; j++) {
                a += Vw[j] * wi[j]; // for efficiency use Vols directly for now
            }
            a += this.biases.w[i];
            A.w[i] = a;
        }
        this.out_act = A;
        return this.out_act;
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        final List<ParamsAndGrads> out = new ArrayList<>();
        for (int i = 0; i < this.out_depth; i++) {
            final ParamsAndGrads pg = new ParamsAndGrads();
            pg.params = this.filters.get(i).w;
            pg.grads = this.filters.get(i).dw;
            pg.l1DecayMul = this.l1_decay_mul;
            pg.l2DecayMul = this.l2_decay_mul;
            out.add(pg);
        }

        final ParamsAndGrads pg = new ParamsAndGrads();
        pg.params = this.biases.w;
        pg.grads = this.biases.dw;
        pg.l1DecayMul = 0;
        pg.l2DecayMul = 0;
        out.add(pg);

        return out.toArray(ParamsAndGrads[]::new);
    }

    @Override
    public JsonObject toJSON() {
        final JsonObject json = new JsonObject();
        json.addProperty("out_depth", this.out_depth);
        json.addProperty("out_sx", this.out_sx);
        json.addProperty("out_sy", this.out_sy);
        json.addProperty("type", this.type.toString());
        json.addProperty("numInputs", this.numInputs);
        json.addProperty("l1_decay_mul", this.l1_decay_mul);
        json.addProperty("l2_decay_mul", this.l2_decay_mul);

        final JsonArray jsonFilters = new JsonArray();
        for (final Vol filter : this.filters) {
            jsonFilters.add(filter.toJSON());
        }
        json.add("filters", jsonFilters);
        json.add("biases", this.biases.toJSON());
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.out_depth = json.get("out_depth").getAsInt();
        this.out_sx = json.get("out_sx").getAsInt();
        this.out_sy = json.get("out_sy").getAsInt();
        this.type = LayerType.valueOf(json.get("type").getAsString());
        this.numInputs = json.get("numInputs").getAsInt();
        this.l1_decay_mul = Double.isNaN(json.get("l1_decay_mul").getAsDouble()) ? 1 : json.get("l1_decay_mul").getAsDouble();
        this.l2_decay_mul = Double.isNaN(json.get("l2_decay_mul").getAsDouble()) ? 1 : json.get("l2_decay_mul").getAsDouble();
        this.filters = new ArrayList<>();
        final JsonArray jsonFilters = json.get("filters").getAsJsonArray();
        for (var i = 0; i < jsonFilters.size(); i++) {
            this.filters.add(Vol.fromJSON(jsonFilters.get(i).getAsJsonObject()));
        }
        this.biases = Vol.fromJSON(json.get("biases").getAsJsonObject());
    }

    @Override
    public double backward(final Vol output) {
        final var V = this.in_act;
        V.dw = Utils.zerosDouble(V.w.length); // zero out the gradient in input Vol

        // compute gradient wrt weights and data
        for (int i = 0; i < this.out_depth; i++) {
            final Vol tfi = this.filters.get(i);
            final double chain_grad = this.out_act.dw[i];
            for (int j = 0; j < this.numInputs; j++) {
                V.dw[j] += tfi.w[j] * chain_grad; // grad wrt input data
                tfi.dw[j] += V.w[j] * chain_grad; // grad wrt params
            }
            this.biases.dw[i] += chain_grad;
        }
        return 0;
    }
}
