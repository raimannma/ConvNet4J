import com.google.gson.JsonObject;

public class MaxoutLayer extends Layer {
    private int[] switches;
    private int groupSize;

    MaxoutLayer() {
        this(new LayerConfig());
    }

    MaxoutLayer(final LayerConfig opt) {
        // required
        this.groupSize = LayerConfig.getOrDefault(2, opt.getGroupSize());

        //computed
        this.out_sx = opt.getInSX();
        this.out_sy = opt.getInSY();
        this.out_depth = (int) Math.floor((double) opt.getInDepth() / this.groupSize);
        this.type = LayerType.MAXOUT;
        this.switches = Utils.zerosInt(this.out_sx * this.out_sy * this.out_depth);
    }

    @Override
    public Vol forward(final Vol v, final boolean isTraining) {
        this.in_act = v;
        final Vol V2 = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

        // optimization branch. If we're operating on 1D arrays we dont have
        // to worry about keeping track of x,y,d coordinates inside
        // input volumes. In convnets we do :(
        if (this.out_sx == 1 && this.out_sy == 1) {
            for (int i = 0; i < this.out_depth; i++) {
                final int ix = i * this.groupSize; // base index offset
                double a = v.w[ix];
                int ai = 0;
                for (int j = 1; j < this.groupSize; j++) {
                    final double a2 = v.w[ix + j];
                    if (a2 > a) {
                        a = a2;
                        ai = j;
                    }
                }
                V2.w[i] = a;
                this.switches[i] = ix + ai;
            }
        } else {
            int n = 0; // counter for switches
            for (int x = 0; x < v.sx; x++) {
                for (int y = 0; y < v.sy; y++) {
                    for (int i = 0; i < this.out_depth; i++) {
                        final int ix = i * this.groupSize;
                        double a = v.get(x, y, ix);
                        int ai = 0;
                        for (int j = 1; j < this.groupSize; j++) {
                            final double a2 = v.get(x, y, ix + j);
                            if (a2 > a) {
                                a = a2;
                                ai = j;
                            }
                        }
                        V2.set(x, y, i, a);
                        this.switches[n] = ix + ai;
                        n++;
                    }
                }
            }

        }
        this.out_act = V2;
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
        json.addProperty("groupSize", this.groupSize);
        json.addProperty("type", this.type.toString());
        return json;
    }

    @Override
    public void fromJSON(final JsonObject json) {
        this.out_depth = json.get("out_depth").getAsInt();
        this.out_sx = json.get("out_sx").getAsInt();
        this.out_sy = json.get("out_sy").getAsInt();
        this.groupSize = json.get("groupSize").getAsInt();
        this.type = LayerType.valueOf(json.get("layer_type").getAsString());
        this.switches = Utils.zerosInt(this.groupSize);
    }

    @Override
    public double backward(final Vol output) {
        final Vol vol = this.in_act; // we need to set dw of this
        final Vol vol2 = this.out_act;
        vol.dw = Utils.zerosDouble(vol.w.length); // zero out gradient wrt data

        // pass the gradient through the appropriate switch
        if (this.out_sx == 1 && this.out_sy == 1) {
            for (var i = 0; i < this.out_depth; i++) {
                vol.dw[this.switches[i]] = vol2.dw[i];
            }
        } else {
            // bleh okay, lets do this the hard way
            var n = 0; // counter for switches
            for (int x = 0; x < vol2.sx; x++) {
                for (int y = 0; y < vol2.sy; y++) {
                    for (int i = 0; i < this.out_depth; i++) {
                        vol.setGrad(x, y, this.switches[n], vol2.getGrad(x, y, i));
                        n++;
                    }
                }
            }
        }
        return 0;
    }
}
