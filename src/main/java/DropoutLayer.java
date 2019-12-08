import com.google.gson.JsonObject;

public class DropoutLayer extends Layer {
    DropoutLayer() {
        this(new LayerConfig());
    }

    DropoutLayer(final LayerConfig opt) {
        super();
    }

    @Override
    public Vol forward(final Vol vol, final boolean isTraining) {
        return null;
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0];
    }

    @Override
    public JsonObject toJSON() {
        return null;
    }

    @Override
    public void fromJSON(final JsonObject jsonLayer) {

    }

    @Override
    public double backward(final Vol output) {
        return 0;
    }
}
