package learning;

import enums.ActivationType;
import layers.Layer;
import layers.LayerConfig;
import utils.Utils;
import utils.Vol;
import utils.Window;

import java.util.*;

public class DQN {
    private final int temporalWindow;
    private final int experienceSize;
    private final int startLearnThreshold;
    private final double gamma;
    private final int learningStepsTotal;
    private final int learningStepsBurnIn;
    private final double epsilonMin;
    private final double epsilonTestTime;
    private final int netInputs;
    private final int numStates;
    private final int numActions;
    private final int windowSize;
    private final Window<Double[]> stateWindow;
    private final Window<Integer> actionWindow;
    private final Window<Double> rewardWindow;
    private final Window<Double[]> netWindow;
    private final Trainer tdTrainer;
    private final ArrayList<Experience> experience;
    private final boolean learning;
    private final Network network;
    private final int batchSize;
    private int age;
    private double epsilon;
    private int forwardPasses;

    DQN(final int numStates, final int numActions, final Map<Option, Double> options) {
        this.temporalWindow = (int) (double) options.getOrDefault(Option.TEMPORAL_WINDOW, 1.0);
        this.experienceSize = (int) (double) options.getOrDefault(Option.EXPERIENCE_SIZE, 30000.0);
        this.startLearnThreshold = (int) (double) options.getOrDefault(Option.START_LEARN_THRESHOLD, Math.floor(Math.min(this.experienceSize * 0.1, 1000)));
        this.gamma = options.getOrDefault(Option.GAMMA, 0.8);
        this.learningStepsTotal = (int) (double) options.getOrDefault(Option.LEARNING_STEPS_TOTAL, 100000.0);
        this.learningStepsBurnIn = (int) (double) options.getOrDefault(Option.LEARNING_STEPS_BURNIN, 3000.0);
        this.epsilonMin = options.getOrDefault(Option.EPSILON_MIN, 0.05);
        this.epsilonTestTime = options.getOrDefault(Option.EPSILON_TEST_TIME, 1.0);

        final int hiddenLayers = (int) (double) options.getOrDefault(Option.HIDDEN_LAYERS, 4.0);

        this.netInputs = numStates * this.temporalWindow + numActions * this.temporalWindow + numStates;
        this.numStates = numStates;
        this.numActions = numActions;
        this.windowSize = Math.max(this.temporalWindow, 2);
        this.stateWindow = new Window<>(this.windowSize);
        this.actionWindow = new Window<>(this.windowSize);
        this.rewardWindow = new Window<>(this.windowSize);
        this.netWindow = new Window<>(this.windowSize);

        final List<LayerConfig> config = new ArrayList<>();

        final LayerConfig input = new LayerConfig();
        input.setType(Layer.LayerType.INPUT);
        input.setOutSX(1);
        input.setOutSY(1);
        input.setOutDepth(this.netInputs);
        config.add(input);

        for (int i = 0; i < hiddenLayers; i++) {
            final LayerConfig hiddenLayer = new LayerConfig();
            hiddenLayer.setType(Layer.LayerType.FC);
            hiddenLayer.setNumNeurons(20);
            hiddenLayer.setActivation(ActivationType.RELU);
            config.add(hiddenLayer);
        }

        final LayerConfig output = new LayerConfig();
        output.setType(Layer.LayerType.REGRESSION);
        output.setNumNeurons(numActions);
        config.add(output);

        this.network = new Network(config.toArray(LayerConfig[]::new));

        this.batchSize = 64;

        final TrainerOptions trainerOptions = new TrainerOptions();
        trainerOptions.learningRate = 0.01;
        trainerOptions.momentum = 0;
        trainerOptions.batchSize = this.batchSize;
        trainerOptions.l2Decay = 0.01;

        this.tdTrainer = new Trainer(this.network, trainerOptions);

        this.experience = new ArrayList<>();
        this.age = 0;
        this.forwardPasses = 0;
        this.epsilon = 1;
        this.learning = true;
    }

    int forward(final double[] input) {
        this.forwardPasses++;

        final int action;
        final Double[] netInput;
        if (this.forwardPasses > this.temporalWindow) {
            netInput = this.getNetInput(Arrays.stream(input).boxed().toArray(Double[]::new));
            if (this.learning) {
                this.epsilon = Math.min(1.0, Math.max(this.epsilonMin, 1.0 - (double) (this.age - this.learningStepsBurnIn) / (this.learningStepsTotal - this.learningStepsBurnIn)));
            } else {
                this.epsilon = this.epsilonTestTime;
            }
            final double rand = Math.random();
            action = rand > this.epsilon
                    ? (int) this.policy(netInput)[0]
                    : this.randomAction();
        } else {
            netInput = new Double[0];
            action = this.randomAction();
        }

        this.netWindow.add(netInput);
        this.stateWindow.add(Arrays.stream(input).boxed().toArray(Double[]::new));
        this.actionWindow.add(action);

        return action;
    }

    private Double[] getNetInput(final Double[] xt) {
        final List<Double> w = new ArrayList<>(Arrays.asList(xt));
        final int n = this.windowSize;
        for (int i = 0; i < this.temporalWindow; i++) {
            w.addAll(Arrays.asList(this.stateWindow.get(n - 1 - i)));

            final Double[] action1ofk = new Double[this.numActions];
            Arrays.fill(action1ofk, 0.0);
            action1ofk[(int) (double) this.actionWindow.get(n - 1 - i)] = (double) this.numStates;
            w.addAll(Arrays.asList(action1ofk));
        }
        return w.toArray(Double[]::new);
    }

    private double[] policy(final Double[] state) {
        final Vol stateVol = new Vol(1, 1, this.netInputs);
        stateVol.w = Arrays.stream(state).mapToDouble(i -> i).toArray();

        final Vol actionValues = this.network.forward(stateVol);
        final double[] doubles = Objects.requireNonNull(Utils.getMaxMin(actionValues.w));
        return new double[]{doubles[0], doubles[1]};
    }

    private int randomAction() {
        return Utils.randInt(0, this.numActions);
    }

    double backward(final double reward) {
        if (!this.learning) {
            return reward;
        }
        this.age++;
        if (this.forwardPasses > this.temporalWindow + 1) {
            final Experience e = new Experience();
            final int n = this.windowSize;
            e.lastState = this.netWindow.get(n - 2);
            e.lastAction = this.actionWindow.get(n - 2);
            e.lastReward = this.rewardWindow.get(n - 2);
            e.state = this.netWindow.get(n - 1);
            if (this.experience.size() < this.experienceSize) {
                this.experience.add(e);
            } else {
                // replace. finite memory!
                this.experience.set(Utils.randInt(0, this.experienceSize), e);
            }
        }
        // learn based on experience, once we have some samples to go on
        // this is where the magic happens...
        if (this.experience.size() > this.startLearnThreshold) {
            double avgCost = 0.0;
            for (int k = 0; k < this.batchSize; k++) {
                final int randInt = Utils.randInt(0, this.experience.size());
                final Experience exp = this.experience.get(randInt);

                final Vol x = new Vol(1, 1, this.netInputs);
                x.w = Arrays.stream(exp.lastState).mapToDouble(i -> i).toArray();

                final double r = exp.lastReward + this.gamma * this.policy(exp.state)[1];

                final Vol yStruct = new Vol(1, 1, 2, 0);
                yStruct.set(0, 0, 0, exp.lastAction);
                yStruct.set(0, 0, 1, r);

                final Map<String, Double> loss = this.tdTrainer.train(x, yStruct);
                avgCost += loss.get("loss");
            }
            avgCost = avgCost / this.batchSize;
            return avgCost;
        }
        return Double.NaN;
    }

    private enum Option {
        EXPERIENCE_SIZE, START_LEARN_THRESHOLD, GAMMA, LEARNING_STEPS_TOTAL, LEARNING_STEPS_BURNIN, EPSILON_MIN, EPSILON_TEST_TIME, HIDDEN_LAYERS, TEMPORAL_WINDOW
    }

    private static final class Experience {
        private Double[] lastState;
        private Double[] state;
        private int lastAction;
        private double lastReward;
    }
}
