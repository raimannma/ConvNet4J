package learning;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.time.Duration.ofSeconds;
import static org.junit.jupiter.api.Assertions.assertTimeout;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DQNTest {
    @Test
    public void testLearningCapabilities() {
        assertTimeout(ofSeconds(5), () -> {
            final Map<DQN.Option, Double> options = new HashMap<>();
            options.put(DQN.Option.EXPERIENCE_SIZE, 100.0);
            options.put(DQN.Option.START_LEARN_THRESHOLD, 1.0);
            options.put(DQN.Option.GAMMA, 0.0);
            options.put(DQN.Option.EPSILON, 0.2);
            final DQN agent = new DQN(1, 2, options);

            double currentState = 0.5;
            double lastState = currentState;
            final int windowSize = 50;
            final List<Double> rewardWindow = new ArrayList<>();
            int i = 0;
            for (i = 0; i < windowSize || rewardWindow.stream().mapToDouble(val -> val).average().orElseThrow() < 0.95; i++) {
                final int action = agent.forward(new double[]{currentState});
                currentState = action == 1 ?
                        Math.min(1, currentState + 0.5) :
                        Math.max(0, currentState - 0.5);

                final double reward = currentState != lastState ? 1 : -1;
                agent.backward(reward);

                rewardWindow.add(reward);
                if (rewardWindow.size() > windowSize) {
                    rewardWindow.remove(0);
                }

                lastState = currentState;
            }
            assertTrue(rewardWindow.stream().mapToDouble(val -> val).average().orElseThrow() >= 0.95);
        });
    }
}