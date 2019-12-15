package learning;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class DQNTest {
    @Test
    public void testLearningCapabilities() {
        final Map<DQN.Option, Double> options = new HashMap<>();
        options.put(DQN.Option.TEMPORAL_WINDOW, 10.0);
        options.put(DQN.Option.START_LEARN_THRESHOLD, 10.0);
        options.put(DQN.Option.GAMMA, 0.1);
        final DQN agent = new DQN(1, 2, options);

        double currentState = 0.5;
        double lastState = currentState;
        double currentLoss;
        final int windowSize = 50;
        final List<Double> rewardWindow = new ArrayList<>();
        double rewardSum = 0;

        for (int i = 0; i < windowSize || rewardSum / windowSize < 0.9; i++) {
            final int action = agent.forward(new double[]{currentState});
            currentState = action == 1 ?
                    Math.min(1, currentState + 0.5) :
                    Math.max(0, currentState - 0.5);

            final double reward = currentState == lastState ? -1 : 1;
            currentLoss = agent.backward(reward);

            System.out.println(currentLoss);

            rewardWindow.add(reward);
            rewardSum += reward;
            if (rewardWindow.size() > windowSize) {
                rewardSum -= rewardWindow.remove(0);
            }

            lastState = currentState;
        }

        assertTrue(rewardSum / windowSize >= 0.9);
    }
}