import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static Layer.LayerType;
import static Layer.LayerType.*;

class Network {
    private List<Layer> layers;

    private Network() {
        this.layers = new ArrayList<>();
    }

    Network(final LayerConfig[] layerConfigs) {
        this.makeLayers(layerConfigs);
    }

    private void makeLayers(final LayerConfig[] defs) {
        if (defs.length < 2) {
            throw new RuntimeException("Error! At least one input layer and one loss layer are required.");
        }
        if (defs[0].type != LayerType.INPUT) {
            throw new RuntimeException("Error! First layer must be the input layer, to declare size of inputs");
        }

        final List<LayerConfig> layerConfigs = new ArrayList<>();
        for (final LayerConfig def : defs) {
            if (def.type == LayerType.SOFTMAX || def.type == LayerType.SVM) {
                final LayerConfig config = new LayerConfig();
                config.setType(LayerType.FC);
                config.setNumNeurons(def.numClasses);
                layerConfigs.add(config);
            }

            if (def.type == LayerType.REGRESSION) {
                final LayerConfig config = new LayerConfig();
                config.setType(LayerType.FC);
                config.setNumNeurons(def.numNeurons);
                layerConfigs.add(config);
            }

            if ((def.type == FC || def.type == CONVOLUTIONAL) && Double.isNaN(def.bias_pref)) {
                def.bias_pref = 0.0;
                if (def.activation != null && def.activation == ActivationType.RELU) {
                    def.bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                    // and will never get any gradient and never contribute any computation. Dead relu.
                }
            }

            layerConfigs.add(def);

            if (def.activation != null) {
                if (def.activation == ActivationType.RELU) {
                    final LayerConfig config = new LayerConfig();
                    config.setType(RELU);
                    layerConfigs.add(config);
                } else if (def.activation == ActivationType.SIGMOID) {
                    final LayerConfig config = new LayerConfig();
                    config.setType(LayerType.SIGMOID);
                    layerConfigs.add(config);
                } else if (def.activation == ActivationType.TANH) {
                    final LayerConfig config = new LayerConfig();
                    config.setType(LayerType.TANH);
                    layerConfigs.add(config);
                } else if (def.activation == ActivationType.MAXOUT) {
                    final LayerConfig config = new LayerConfig();
                    config.setType(MAXOUT);
                    config.setGroupSize(Double.isNaN(def.group_size) ? 2 : def.group_size);
                    layerConfigs.add(config);
                } else {
                    throw new RuntimeException("ERROR unsupported activation " + def.activation.toString());
                }
            }
            if (!Double.isNaN(def.drop_prob) && def.type != DROPOUT) {
                final LayerConfig config = new LayerConfig();
                config.setType(DROPOUT);
                config.setDropProb(def.drop_prob);
                layerConfigs.add(config);
            }
        }

        this.layers = new ArrayList<>();

        for (int i = 0; i < layerConfigs.size(); i++) {
            final LayerConfig def = layerConfigs.get(i);
            if (i > 0) {
                final Layer prev = this.layers.get(i - 1);
                def.in_sx = prev.out_sx;
                def.in_sy = prev.out_sy;
                def.in_depth = prev.out_depth;
            }

            switch (def.type) {
                case FC:
                    this.layers.add(new FullyConnLayer(def));
                    break;
                case LRN:
                    this.layers.add(new LocalResponseNormalizationLayer(def));
                    break;
                case DROPOUT:
                    this.layers.add(new DropoutLayer(def));
                    break;
                case INPUT:
                    this.layers.add(new InputLayer(def));
                    break;
                case SOFTMAX:
                    this.layers.add(new SoftmaxLayer(def));
                    break;
                case REGRESSION:
                    this.layers.add(new RegressionLayer(def));
                    break;
                case CONVOLUTIONAL:
                    this.layers.add(new ConvolutionalLayer(def));
                    break;
                case POOL:
                    this.layers.add(new PoolLayer(def));
                    break;
                case RELU:
                    this.layers.add(new ReluLayer(def));
                    break;
                case SIGMOID:
                    this.layers.add(new SigmoidLayer(def));
                    break;
                case TANH:
                    this.layers.add(new TanhLayer(def));
                    break;
                case MAXOUT:
                    this.layers.add(new MaxoutLayer(def));
                    break;
                case SVM:
                    this.layers.add(new SVMLayer(def));
                    break;
                default:
                    throw new RuntimeException("ERROR: UNRECOGNIZED LAYER TYPE: " + def.type))
            }
        }
    }

    static Network fromJSON(final JsonObject json) {
        final Network network = new Network();
        final JsonArray jsonLayers = json.get("layers").getAsJsonArray();
        for (int i = 0; i < jsonLayers.size(); i++) {
            final JsonObject jsonLayer = jsonLayers.get(i).getAsJsonObject();
            final LayerType type = LayerType.valueOf(jsonLayer.get("type").getAsString());
            final Layer layer;
            switch (type) {
                case FC:
                    layer = new FullyConnectedLayer();
                    break;
                case LRN:
                    layer = new LocalResponseNormalizationLayer();
                    break;
                case DROPOUT:
                    layer = new DropoutLayer();
                    break;
                case SOFTMAX:
                    layer = new SoftmaxLayer();
                    break;
                case REGRESSION:
                    layer = new RegressionLayer();
                    break;
                case CONVOLUTIONAL:
                    layer = new ConvolutionalLayer();
                    break;
                case POOL:
                    layer = new PoolLayer();
                    break;
                case RELU:
                    layer = new ReluLayer();
                    break;
                case SIGMOID:
                    layer = new SigmoidLayer();
                    break;
                case TANH:
                    layer = new TanhLayer();
                    break;
                case MAXOUT:
                    layer = new MaxoutLayer();
                    break;
                case INPUT:
                    layer = new InputLayer();
                    break;
                case SVM:
                    layer = new SVMLayer();
                    break;
                default:
                    throw new IllegalStateException("Unexpected value: " + type);
            }
            layer.fromJSON(jsonLayer);
            network.layers.add(layer);
        }
        return network;
    }

    JsonObject toJSON() {
        final JsonObject json = new JsonObject();

        final JsonArray jsonLayers = new JsonArray();
        for (final Layer layer : this.layers) {
            jsonLayers.add(layer.toJSON());
        }

        json.add("layers", jsonLayers);
        return json;
    }

    public ParamsAndGrads[] getParamsAndGrads() {
        final List<ParamsAndGrads> out = new ArrayList<>();
        for (final Layer layer : this.layers) {
            out.addAll(Arrays.asList(layer.getParamsAndGrads()));
        }
        return out.toArray(ParamsAndGrads[]::new);
    }

    public double backward(final Vol output) {
        final double loss = this.layers.get(this.layers.size() - 1).backward(output);
        for (int i = this.layers.size() - 2; i >= 0; i--) {
            this.layers.get(i).backward();
        }
        return loss;
    }

    public void forward(final Vol input) {
        this.forward(input, false);
    }

    public Vol forward(final Vol vol, final boolean isTraining) {
        Vol act = vol;
        for (final Layer layer : this.layers) {
            act = layer.forward(act, isTraining);
        }
        return act;
    }

    double getCostLoss(final Vol vol, final Vol output) {
        this.forward(vol, false);
        return this.layers.get(this.layers.size() - 1).backward(output);
    }

    int getPrediction() {
        final Layer lastLayer = this.layers.get(this.layers.size() - 1);
        if (lastLayer.type != LayerType.SOFTMAX) {
            throw new RuntimeException("ERROR: getPrediction function assumes softmax as last layer of the net!");
        }

        final double[] p = lastLayer.out_act.w;
        double maxv = p[0];
        int maxi = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > maxv) {
                maxv = p[i];
                maxi = i;
            }
        }
        return maxi;
    }
}
