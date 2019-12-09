package learning;

import enums.ActivationType;
import layers.Layer;
import layers.LayerConfig;
import org.junit.jupiter.api.Test;
import utils.Vol;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NetworkTest {
    private static Network network;
    private static Trainer trainer;

    @Test
    public void testInitialization() {
        NetworkTest.initNet();
        assertEquals(7, network.layers.size());
    }

    private static void initNet() {
        final List<LayerConfig> configs = new ArrayList<>();

        final LayerConfig input = new LayerConfig();
        input.setType(Layer.LayerType.INPUT);
        input.setOutSX(1);
        input.setOutSY(1);
        input.setOutDepth(2);
        configs.add(input);

        final LayerConfig hidden = new LayerConfig();
        hidden.setType(Layer.LayerType.FC);
        hidden.setNumNeurons(5);
        hidden.setActivation(ActivationType.TANH);
        configs.add(hidden);

        final LayerConfig hidden2 = new LayerConfig();
        hidden2.setType(Layer.LayerType.FC);
        hidden2.setNumNeurons(5);
        hidden2.setActivation(ActivationType.TANH);
        configs.add(hidden2);


        final LayerConfig output = new LayerConfig();
        output.setType(Layer.LayerType.SOFTMAX);
        output.setNumClasses(3);
        configs.add(output);

        network = new Network(configs.toArray(LayerConfig[]::new));

        final TrainerOptions options = new TrainerOptions();
        options.setLearningRate(0.0001);
        options.setMomentum(0);
        options.setBatchSize(1);
        options.setL2Decay(0);
        trainer = new Trainer(network, options);
    }

    @Test
    public void testForwardPropagation() {
        initNet();

        final Vol x = new Vol(new double[]{0.2, -0.3});
        final Vol probabilityVolume = network.forward(x);
        assertEquals(3, probabilityVolume.w.length);

        double sum = 0;
        for (int i = 0; i < probabilityVolume.w.length; i++) {
            assertTrue(probabilityVolume.w[i] > 0);
            assertTrue(probabilityVolume.w[i] < 1);
            sum += probabilityVolume.w[i];
        }
        assertTrue(Math.abs(sum - 1) < 0.001);
    }

    @Test
    public void testLearning() {
        initNet();

        for (int i = 0; i < 100; i++) {
            final Vol x = new Vol(new double[]{Math.random() * 2 - 1, Math.random() * 2 - 1});
            final Vol pv = network.forward(x);
            final int groundTruthIndex = (int) Math.floor(Math.random() * 3);
            final Vol output = new Vol(1, 1, 1, groundTruthIndex);
            trainer.train(x, output);
            final Vol pv2 = network.forward(x);
            assertTrue(pv2.w[groundTruthIndex] >= pv.w[groundTruthIndex]);
        }
    }

    @Test
    public void testGradiantComputing() {
        initNet();

        final Vol x = new Vol(new double[]{Math.random() * 2 - 1, Math.random() * 2 - 1});
        final Vol output = new Vol(1, 1, 1, Math.floor(Math.random() * 3));
        trainer.train(x, output); // computes gradients at all layers, and at x

        final double delta = 0.000001;

        for (int i = 0; i < x.w.length; i++) {
            final double temp = x.w[i];

            x.w[i] += delta;
            final double c0 = Math.abs(network.getCostLoss(x, output));
            x.w[i] -= 2 * delta;
            final double c1 = Math.abs(network.getCostLoss(x, output));
            x.w[i] = temp; // reset

            final double gradNumeric = (c0 - c1) / (2 * delta);
            final double relError = (x.dw[i] - gradNumeric) / (x.dw[i] + gradNumeric);

            assertTrue(Math.abs(relError) < 0.1);
        }
    }
}