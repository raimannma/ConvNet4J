import java.util.ArrayList;
import java.util.List;

public class Trainer {

    private final Network net;
    private final double learningRate;
    private final double l1Decay;
    private final double l2Decay;
    private final double batchSize;
    private final double momentum;
    private final double ro;
    private final double epsilon;
    private final double beta1;
    private final double beta2;
    private final List<List<Double>> gSum;
    private final List<List<Double>> xSum;
    private final boolean regression;
    private final TrainerMethod method;
    private double k;

    public Trainer(final Network network, final TrainerOptions options) {
        this.net = network;
        this.learningRate = options.getLearningRate();
        this.l1Decay = options.getL1Decay();
        this.l2Decay = options.getL2Decay();
        this.batchSize = options.getBatchSize();
        this.method = options.getMethod();

        this.momentum = options.getMomentum();
        this.ro = options.getRo();
        this.epsilon = options.getEpsilon();
        this.beta1 = options.getBeta1();
        this.beta2 = options.getBeta2();

        this.k = 0;
        this.gSum = new ArrayList<>();
        this.xSum = new ArrayList<>();

        this.regression = this.net.layers.get(this.net.layers.size() - 1).type == LayerType.REGRESSION;
    }

    public void train(final Vol x, final Vol y) {
        this.net.forward(x, true);

        final double costLoss = this.net.backward(y);
        final double l2DecayLoss = 0;
        final double l1DecayLoss = 0;

        this.k++;
        if (this.k % this.batchSize == 0) {
        }
    }

}
