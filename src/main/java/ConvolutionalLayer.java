import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.util.ArrayList;
import java.util.List;

class ConvolutionalLayer extends Layer {
    private final int in_sx;
    private final int in_sy;
    private Vol biases;
    private int pad;
    private double l1_decay_mul;
    private double l2_decay_mul;
    private ArrayList<Vol> filters;
    private int sx;
    private int sy;
    private int in_depth;
    private int stride;
    private int out_sy;
    private int out_depth;
    private int out_sx;
    private Vol in_act;


    ConvolutionalLayer() {
        this(new LayerConfig());
    }

    ConvolutionalLayer(final LayerConfig opt) {
        //required
        this.out_depth = opt.getFilters();
        this.sx = opt.getSX();
        this.in_depth = opt.getInDepth();
        this.in_sx = opt.getInSX();
        this.in_sy = opt.getInSY();

        //optional
        this.sy = LayerConfig.getOrDefault(opt.getSY(), this.sx);
        this.stride = LayerConfig.getOrDefault(opt.getStride(), 1); // stride at which we apply filters to input volume
        this.pad = LayerConfig.getOrDefault(opt.getPad(), 0); // amount of 0 padding to add around borders of input volume
        this.l1_decay_mul = LayerConfig.getOrDefault(0, opt.getL1DecayMul());
        this.l2_decay_mul = LayerConfig.getOrDefault(1, opt.getL2DecayMul());

        // computed
        // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        this.out_sx = (int) Math.floor((double) (this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
        this.out_sy = (int) Math.floor((double) (this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
        this.type = LayerType.CONVOLUTIONAL;

        final double bias = LayerConfig.getOrDefault(0, opt.getBiasPref());
        this.filters = new ArrayList<>();
        for (int i = 0; i < this.out_depth; i++) {
            this.filters.add(new Vol(this.sx, this.sy, this.in_depth));
        }
        this.biases = new Vol(1, 1, this.out_depth, bias);
    }


    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.in_act = vol;
        final Vol A = new Vol(
                Double.isNaN(this.out_sx) ? 0 : this.out_sx,
                Double.isNaN(this.out_sy) ? 0 : this.out_sy,
                Double.isNaN(this.out_depth) ? 0 : this.out_depth,
                0);

        final int V_sx = Double.isNaN(vol.sx) ? 0 : vol.sx;
        final int V_sy = Double.isNaN(vol.sy) ? 0 : vol.sy;
        final double xyStride = Double.isNaN(this.stride) ? 0 : this.stride;

        for (int d = 0; d < this.out_depth; d++) {
            final Vol f = this.filters.get(d);
            int y = Double.isNaN(this.pad) ? 0 : -this.pad;
            for (int ay = 0; ay < this.out_sy; y += xyStride, ay++) {  // xy_stride
                int x = Double.isNaN(this.pad) ? 0 : -this.pad;
                for (int ax = 0; ax < this.out_sx; x += xyStride, ax++) {  // xy_stride
                    // convolve centered at this particular location
                    double a = 0.0;
                    for (int fy = 0; fy < f.sy; fy++) {
                        final int oy = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < f.sx; fx++) {
                            final int ox = x + fx;
                            if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
                                for (int fd = 0; fd < f.depth; fd++) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    a += f.w[((f.sx * fy) + fx) * f.depth + fd] * vol.w[((V_sx * oy) + ox) * vol.depth + fd];
                                }
                            }
                        }
                    }
                    a += this.biases.w[d];
                    A.set(ax, ay, d, a);
                }
            }
        }
        this.out_act = A;
        return A;
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
        final ParamsAndGrads paramsAndGrads = new ParamsAndGrads();
        paramsAndGrads.params = this.biases.w;
        paramsAndGrads.grads = this.biases.dw;
        paramsAndGrads.l1DecayMul = 0;
        paramsAndGrads.l2DecayMul = 0;
        out.add(paramsAndGrads);

        return out.toArray(ParamsAndGrads[]::new);
    }

    @Override
    JsonObject toJSON() {
        final JsonObject json = new JsonObject();
        json.addProperty("sx", this.sx);
        json.addProperty("sy", this.sy);
        json.addProperty("stride", this.stride);
        json.addProperty("in_depth", this.in_depth);
        json.addProperty("out_depth", this.out_depth);
        json.addProperty("out_sx", this.out_sx);
        json.addProperty("out_sy", this.out_sy);
        json.addProperty("type", this.type.toString());
        json.addProperty("l1_decay_mul", this.l1_decay_mul);
        json.addProperty("l2_decay_mul", this.l2_decay_mul);
        json.addProperty("pad", this.pad);

        final JsonArray jsonFilters = new JsonArray();
        for (final Vol filter : this.filters) {
            jsonFilters.add(filter.toJSON());
        }
        json.add("filters", jsonFilters);
        json.add("biases", this.biases.toJSON());
        return json;
    }

    @Override
    void fromJSON(final JsonObject json) {
        this.out_depth = json.get("out_depth").getAsInt();
        this.out_sx = json.get("out_sx").getAsInt();
        this.out_sy = json.get("out_sy").getAsInt();
        this.type = LayerType.valueOf(json.get("type").getAsString());
        this.sx = json.get("sx").getAsInt(); // filter size in x, y dims
        this.sy = json.get("sy").getAsInt();
        this.stride = json.get("stride").getAsInt();
        this.in_depth = json.get("in_depth").getAsInt(); // depth of input volume
        this.filters = new ArrayList<>();
        this.l1_decay_mul = Double.isNaN(json.get("l1_decay_mul").getAsDouble()) ? 1 : json.get("l1_decay_mul").getAsDouble();
        this.l2_decay_mul = Double.isNaN(json.get("l2_decay_mul").getAsDouble()) ? 1 : json.get("l2_decay_mul").getAsDouble();
        this.pad = Double.isNaN(json.get("pad").getAsDouble()) ? 0 : json.get("pad").getAsInt();
        final JsonArray jsonFilters = json.get("filters").getAsJsonArray();
        for (var i = 0; i < jsonFilters.size(); i++) {
            this.filters.add(Vol.fromJSON(jsonFilters.get(i).getAsJsonObject()));
        }
        this.biases = Vol.fromJSON(json.get("biases").getAsJsonObject());
    }

    @Override
    public double backward(final Vol output) {
        final var V = this.in_act;
        V.dw = Utils.zerosDouble(V.w.length); // zero out gradient wrt bottom data, we're about to fill it

        final var V_sx = Double.isNaN(V.sx) ? 0 : V.sx;
        final var V_sy = Double.isNaN(V.sy) ? 0 : V.sy;
        final var xy_stride = Double.isNaN(this.stride) ? 0 : this.stride;

        for (int d = 0; d < this.out_depth; d++) {
            final Vol f = this.filters.get(d);
            int y = Double.isNaN(this.pad) ? 0 : -this.pad;
            for (int ay = 0; ay < this.out_sy; y += xy_stride, ay++) {  // xy_stride
                int x = Double.isNaN(this.pad) ? 0 : -this.pad;
                for (int ax = 0; ax < this.out_sx; x += xy_stride, ax++) {  // xy_stride
                    // convolve centered at this particular location
                    final double chain_grad = this.out_act.getGrad(ax, ay, d); // gradient from above, from chain rule
                    for (int fy = 0; fy < f.sy; fy++) {
                        final int oy = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < f.sx; fx++) {
                            final int ox = x + fx;
                            if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
                                for (int fd = 0; fd < f.depth; fd++) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    final int ix1 = ((V_sx * oy) + ox) * V.depth + fd;
                                    final int ix2 = ((f.sx * fy) + fx) * f.depth + fd;
                                    f.dw[ix2] += V.w[ix1] * chain_grad;
                                    V.dw[ix1] += f.w[ix2] * chain_grad;
                                }
                            }
                        }
                    }
                    this.biases.dw[d] += chain_grad;
                }
            }
        }
        return 0;
    }
}