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
        System.out.println(Math.abs(sum - 1));
    }
}