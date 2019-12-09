package learning;

import enums.ActivationType;
import layers.Layer;
import layers.LayerConfig;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class NetworkTest {
    private static Network network;
    private static Trainer trainer;

    @Test
    public void testInitialization() {
        NetworkTest.initNet();
        System.out.println(network.layers.toString());
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
        //add it 2x
        configs.add(hidden);
        configs.add(hidden);


        final LayerConfig output = new LayerConfig();
        output.setType(Layer.LayerType.SOFTMAX);
        output.setNumClasses(3);
        configs.add(output);

        configs.forEach(layerConfig -> System.out.println(layerConfig.type));

        network = new Network(configs.toArray(LayerConfig[]::new));

        final TrainerOptions options = new TrainerOptions();
        options.setLearningRate(0.0001);
        options.setMomentum(0);
        options.setBatchSize(1);
        options.setL2Decay(0);
        trainer = new Trainer(network, options);
    }
}