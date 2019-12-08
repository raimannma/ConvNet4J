import com.google.gson.JsonObject;

public class PoolLayer extends Layer {
    private final int in_sy;
    private final int in_sx;
    private int sx;
    private int sy;
    private int in_depth;
    private int stride;
    private int pad;
    private int[] switchX;
    private int[] switchY;

    PoolLayer() {
        this(new LayerConfig());
    }

    PoolLayer(final LayerConfig opt) {
        // required
        this.sx = opt.getSX();
        this.in_depth = opt.getInDepth();
        this.in_sx = opt.getInSX();
        this.in_sy = opt.getInSY();

        // optional
        this.sy = LayerConfig.getOrDefault(this.sx, opt.getSY());
        this.stride = LayerConfig.getOrDefault(2, opt.getStride());
        this.pad = LayerConfig.getOrDefault(0, opt.getPad());

        //computed
        this.out_depth = this.in_depth;
        this.out_sx = (int) Math.floor((double) (this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
        this.out_sx = (int) Math.floor((double) (this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
        this.type = LayerType.POOL;

        this.switchX = Utils.zerosInt(this.out_sx * this.out_sy * this.out_depth);
        this.switchY = Utils.zerosInt(this.out_sx * this.out_sy * this.out_depth);
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        this.in_act = vol;

        final Vol activate = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

        int n = 0; // a counter for switches
        for (int d = 0; d < this.out_depth; d++) {
            int x = -this.pad;
            int y;
            for (int ax = 0; ax < this.out_sx; x += this.stride, ax++) {
                y = -this.pad;
                for (int ay = 0; ay < this.out_sy; y += this.stride, ay++) {

                    // convolve centered at this particular location
                    double max = Double.MIN_VALUE; // hopefully small enough ;\
                    int winx = -1;
                    int winy = -1;
                    for (int fx = 0; fx < this.sx; fx++) {
                        for (int fy = 0; fy < this.sy; fy++) {
                            final int oy = y + fy;
                            final int ox = x + fx;
                            if (oy >= 0 && oy < vol.sy && ox >= 0 && ox < vol.sx) {
                                final double v = vol.get(ox, oy, d);
                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop
                                // and can help make nice visualizations in future
                                if (v > max) {
                                    max = v;
                                    winx = ox;
                                    winy = oy;
                                }
                            }
                        }
                    }
                    this.switchX[n] = winx;
                    this.switchY[n] = winy;
                    n++;
                    activate.set(ax, ay, d, max);
                }
            }
        }
        this.out_act = activate;
        return this.out_act;
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0];
    }

    @Override
    public JsonObject toJSON() {
        final JsonObject json = new JsonObject();
        json.addProperty("sx", this.sx);
        json.addProperty("sy", this.sy);
        json.addProperty("stride", this.stride);
        json.addProperty("in_depth", this.in_depth);
        json.addProperty("out_depth", this.out_depth);
        json.addProperty("out_sx", this.out_sx);
        json.addProperty("out_sy", this.out_sy);
        json.addProperty("type", this.type.toString());
        json.addProperty("pad", this.pad);
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.out_depth = json.get("out_depth").getAsInt();
        this.out_sx = json.get("out_sx").getAsInt();
        this.out_sy = json.get("out_sy").getAsInt();
        this.type = LayerType.valueOf(json.get("layer_type").getAsString());
        this.sx = json.get("sx").getAsInt();
        this.sy = json.get("sy").getAsInt();
        this.stride = json.get("stride").getAsInt();
        this.in_depth = json.get("in_depth").getAsInt();
        this.pad = Double.isNaN(json.get("pad").getAsInt()) ? 0 : json.get("pad").getAsInt(); // backwards compatibility
        this.switchX = Utils.zerosInt(this.out_sx * this.out_sy * this.out_depth); // need to re-init these appropriately
        this.switchY = Utils.zerosInt(this.out_sx * this.out_sy * this.out_depth);
    }

    @Override
    public double backward(final Vol output) {
        // pooling layers have no parameters, so simply compute
        // gradient wrt data here
        final Vol vol = this.in_act;
        vol.dw = Utils.zerosDouble(vol.w.length); // zero out gradient wrt data

        var n = 0;
        for (int d = 0; d < this.out_depth; d++) {
            int x = -this.pad;
            for (int ax = 0; ax < this.out_sx; x += this.stride, ax++) {
                int y = -this.pad;
                for (int ay = 0; ay < this.out_sy; y += this.stride, ay++) {
                    final double chain_grad = this.out_act.getGrad(ax, ay, d);
                    vol.addGrad(this.switchX[n], this.switchY[n], d, chain_grad);
                    n++;
                }
            }
        }
        return 0;
    }
}
