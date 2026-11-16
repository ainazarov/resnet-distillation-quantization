package org.fooddiary.benchmark;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONObject;

import java.util.Locale;

public final class MainActivity extends Activity {
    private Button runButton;
    private TextView status;

    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        int pad = Math.round(24 * getResources().getDisplayMetrics().density);
        layout.setPadding(pad, pad, pad, pad);
        layout.setOnApplyWindowInsetsListener((v, insets) -> {
            v.setPadding(pad, pad + insets.getSystemWindowInsetTop(), pad,
                    pad + insets.getSystemWindowInsetBottom());
            return insets;
        });
        TextView title = new TextView(this);
        title.setText(getApplicationInfo().loadLabel(getPackageManager()));
        title.setTextSize(26);
        layout.addView(title);
        runButton = new Button(this);
        runButton.setText("Run benchmark");
        runButton.setOnClickListener(v -> startBenchmark(false));
        layout.addView(runButton, ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        status = new TextView(this);
        status.setTextSize(16);
        status.setText(Benchmark.DISPLAY_NAMES[0] + " and " + Benchmark.DISPLAY_NAMES[1]
                + " · CPU · 4 threads · batch 1");
        layout.addView(status);
        setContentView(layout);
        if (getIntent().getBooleanExtra("benchmark", false))
            startBenchmark(getIntent().getBooleanExtra("stages", false));
    }

    private void startBenchmark(boolean stages) {
        if (!runButton.isEnabled()) return;
        runButton.setEnabled(false);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        status.setText("Loading models and photos…");
        new Thread(() -> {
            String message;
            try {
                JSONObject report = Benchmark.run(this, stages, text -> runOnUiThread(() -> status.setText(text)));
                StringBuilder text = new StringBuilder();
                for (int j = 0; j < Benchmark.MODELS.length; j++) {
                    JSONObject stats = report.getJSONObject("models").getJSONObject(Benchmark.MODELS[j]);
                    text.append(String.format(Locale.US,
                            "%s\nMedian forward %.2f ms\nJPEG to top 3, p95 %.2f ms\n"
                                    + "Top 3 matches Python on %d/%d photos\n\n",
                            Benchmark.DISPLAY_NAMES[j],
                            stats.getDouble("median_forward_ms"), stats.getDouble("p95_jpeg_to_top3_ms"),
                            stats.getInt("ordered_top3_matches"), report.getInt("photos")));
                }
                message = text.toString().trim();
                Log.i("Food101Bench", stages ? "Saved files/stage-profile.json" : "Saved files/benchmark.json");
            } catch (Exception | LinkageError error) {
                Log.e("Food101Bench", "Benchmark failed", error);
                message = "Benchmark failed\n" + error;
            }
            String result = message;
            runOnUiThread(() -> {
                status.setText(result);
                runButton.setEnabled(true);
                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            });
        }, "food101-benchmark").start();
    }
}
