package org.fooddiary.benchmark;

import android.content.Context;
import android.os.Build;
import android.os.Debug;
import android.os.PowerManager;
import android.os.SystemClock;

import org.json.JSONArray;
import org.json.JSONObject;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;

import java.io.File;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;

final class Benchmark {
    static final String[] MODELS = BuildConfig.CONTROL_BENCHMARK ? new String[]{"plain", "distilled"} : BuildConfig.PRECISION_BENCHMARK
            ? new String[]{"resnet18_fp32", "resnet18_int8"} : new String[]{"teacher", "distilled"};
    static final String[] DISPLAY_NAMES = BuildConfig.CONTROL_BENCHMARK ? new String[]{"Plain ResNet-18 FP32", "Distilled ResNet-18 FP32"} : BuildConfig.PRECISION_BENCHMARK
            ? new String[]{"ResNet-18 FP32", "ResNet-18 static INT8"}
            : new String[]{"ResNet-50", "Distilled ResNet-18"};
    private static final int THREADS = 4;
    private static final int WARMUP = 20;

    private static File stage(Context app, String asset, JSONObject hashes) throws Exception {
        // Stage each packaged asset once per pass. Only JPEG file open through label lookup is timed.
        File dest = new File(app.getCacheDir(), asset);
        Files.createDirectories(dest.getParentFile().toPath());
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream input = new DigestInputStream(app.getAssets().open(asset), digest)) {
            Files.copy(input, dest.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
        StringBuilder hex = new StringBuilder(64);
        for (byte b : digest.digest()) hex.append(String.format(Locale.ROOT, "%02x", b & 255));
        hashes.put(asset, hex.toString());
        return dest;
    }

    private static final class Measurement {
        long forwardNs;
        long elapsedNs;
        long decodeNs;
        long prepareNs;
        long rankNs;
        final int[] top3 = new int[3];
        final String[] dishes = new String[3];
    }

    private static Measurement measure(Module module, File photo, List<String> names, boolean stages) throws Exception {
        Measurement m = new Measurement();
        long[] decodedAt = stages ? new long[1] : null;
        long start = SystemClock.elapsedRealtimeNanos();
        EValue input = EValue.from(Preprocess.fromJpeg(photo, decodedAt));
        long beforeForward = SystemClock.elapsedRealtimeNanos();
        EValue[] output = module.forward(input);
        long afterForward = SystemClock.elapsedRealtimeNanos();
        float[] logits = output[0].toTensor().getDataAsFloatArray();
        if (logits.length != names.size()) throw new IllegalStateException("Wrong number of logits");
        Arrays.fill(m.top3, -1);
        for (int i = 0; i < logits.length; i++) {
            if (!Float.isFinite(logits[i])) throw new IllegalStateException("Non-finite logit");
            for (int j = 0; j < m.top3.length; j++) {
                if (m.top3[j] == -1 || logits[i] > logits[m.top3[j]]) {
                    for (int k = m.top3.length - 1; k > j; k--) m.top3[k] = m.top3[k - 1];
                    m.top3[j] = i;
                    break;
                }
            }
        }
        for (int i = 0; i < m.top3.length; i++) m.dishes[i] = names.get(m.top3[i]);
        long end = SystemClock.elapsedRealtimeNanos();
        m.elapsedNs = end - start;
        m.forwardNs = afterForward - beforeForward;
        if (stages) {
            m.decodeNs = decodedAt[0] - start;
            m.prepareNs = beforeForward - decodedAt[0];
            m.rankNs = end - afterForward;
        }
        return m;
    }

    private static double percentileMs(long[] times, double quantile) {
        Arrays.sort(times);
        double index = quantile * (times.length - 1);
        int left = (int) index;
        long right = times[(int) Math.ceil(index)];
        return (times[left] + (right - times[left]) * (index - left)) / 1e6;
    }

    private static void checkDebugger() {
        if (Debug.isDebuggerConnected() || Debug.waitingForDebugger())
            throw new IllegalStateException("Disconnect the debugger and run the benchmark again.");
    }

    static JSONObject run(Context app, boolean stages, Consumer<String> progress) throws Exception {
        checkDebugger();
        JSONObject hashes = new JSONObject();
        File ref = stage(app, "reference.json", hashes);
        JSONObject reference = new JSONObject(new String(Files.readAllBytes(ref.toPath()), StandardCharsets.UTF_8));
        JSONArray photos = reference.getJSONArray("photos");
        if (photos.length() < WARMUP) throw new IllegalArgumentException("Incomplete JPEG manifest");
        File labels = stage(app, "classes_food101.txt", hashes);
        List<String> names = Files.readAllLines(labels.toPath(), StandardCharsets.UTF_8);
        if (names.size() != 101) throw new IllegalArgumentException("Expected 101 Food-101 labels");
        File[] files = new File[photos.length()];
        for (int i = 0; i < files.length; i++) {
            JSONObject photo = photos.getJSONObject(i);
            String path = photo.getString("asset");
            files[i] = stage(app, path, hashes);
            if (!hashes.getString(path).equals(photo.getString("sha256")))
                throw new IllegalStateException("Changed photo " + path);
        }
        File[] exports = new File[MODELS.length];
        for (int j = 0; j < MODELS.length; j++) {
            String asset = "models/" + MODELS[j] + ".pte";
            exports[j] = stage(app, asset, hashes);
            if ((BuildConfig.PRECISION_BENCHMARK || BuildConfig.CONTROL_BENCHMARK) &&
                    !hashes.getString(asset).equals(reference.getJSONObject("pte_sha256").getString(MODELS[j])))
                throw new IllegalStateException("Python reference generated from a different " + asset);
        }
        JSONObject report = new JSONObject();
        report.put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL)
                .put("device", Build.DEVICE).put("android_version", Build.VERSION.RELEASE)
                .put("android_sdk", Build.VERSION.SDK_INT).put("abi", Build.SUPPORTED_ABIS[0]);
        if (Build.VERSION.SDK_INT >= 31) report.put("soc", Build.SOC_MODEL);
        report.put("executorch", "1.5.1").put("threads", THREADS).put("batch_size", 1)
                .put("precision", BuildConfig.PRECISION_BENCHMARK ? "FP32 vs static INT8" : "FP32")
                .put("tensor_io_dtype", "FP32").put("warmup_per_model", WARMUP).put("photos", photos.length())
                .put("reference_sha256", hashes.getString("reference.json"))
                .put("labels_sha256", hashes.getString("classes_food101.txt"))
                .put("file_cache", "warm").put("debugger_attached", false);
        JSONObject stats = new JSONObject();
        JSONArray observations = new JSONArray();
        int[] matches = new int[MODELS.length];
        long[][] forward = new long[MODELS.length][photos.length()];
        long[][] latency = new long[MODELS.length][photos.length()];
        PowerManager power = app.getSystemService(PowerManager.class);
        try (Module first = Module.load(exports[0].getAbsolutePath(), Module.LOAD_MODE_MMAP, THREADS);
             Module second = Module.load(exports[1].getAbsolutePath(), Module.LOAD_MODE_MMAP, THREADS)) {
            Module[] modules = {first, second};
            for (int j = 0; j < modules.length; j++) {
                modules[j].loadMethod("forward");
                String[] backends = modules[j].getMethodMetadata("forward").getBackends();
                if (!Arrays.asList(backends).contains("XnnpackBackend"))
                    throw new IllegalStateException("Expected XNNPACK, found " + Arrays.toString(backends));
                stats.put(MODELS[j], new JSONObject().put("backends", new JSONArray(backends))
                        .put("weight_precision", MODELS[j].equals("resnet18_int8") ? "INT8" : "FP32")
                        .put("pte_bytes", exports[j].length())
                        .put("pte_sha256", hashes.getString("models/" + MODELS[j] + ".pte")));
            }
            progress.accept("Warming up both models…");
            for (int i = 0; i < WARMUP; i++) {
                for (Module model : modules) measure(model, files[i], names, stages);
            }
            report.put("start_unix_ms", System.currentTimeMillis());
            report.put("power_save_start", power.isPowerSaveMode());
            if (Build.VERSION.SDK_INT >= 29) report.put("thermal_status_start", power.getCurrentThermalStatus());
            for (int i = 0; i < files.length; i++) {
                JSONObject photo = photos.getJSONObject(i);
                for (int position = 0; position < MODELS.length; position++) {
                    int j = (i + position) % MODELS.length;
                    Measurement m = measure(modules[j], files[i], names, stages);
                    forward[j][i] = m.forwardNs;
                    latency[j][i] = m.elapsedNs;
                    JSONArray expected = photo.getJSONArray(MODELS[j]);
                    boolean match = true;
                    for (int k = 0; k < 3; k++) match &= expected.getInt(k) == m.top3[k];
                    if (match) matches[j]++;
                    JSONObject row = new JSONObject().put("id", photo.getString("id"))
                            .put("model", MODELS[j]).put("position", position)
                            .put("forward_ms", m.forwardNs / 1e6).put("jpeg_to_top3_ms", m.elapsedNs / 1e6)
                            .put("top3", new JSONArray(m.top3)).put("dishes", new JSONArray(m.dishes))
                            .put("python_top3", expected).put("ordered_match", match);
                    if (stages) row.put("decode_rgb_ms", m.decodeNs / 1e6)
                            .put("resize_crop_norm_tensor_ms", m.prepareNs / 1e6)
                            .put("top3_readback_labels_ms", m.rankNs / 1e6);
                    observations.put(row);
                }
                if ((i + 1) % 20 == 0)
                    progress.accept("Measured " + (i + 1) + " of " + files.length + " photos");
            }
            report.put("power_save_end", power.isPowerSaveMode());
            if (Build.VERSION.SDK_INT >= 29) report.put("thermal_status_end", power.getCurrentThermalStatus());
        }
        checkDebugger();
        for (int j = 0; j < MODELS.length; j++) {
            stats.getJSONObject(MODELS[j]).put("median_forward_ms", percentileMs(forward[j], 0.5))
                    .put("p95_jpeg_to_top3_ms", percentileMs(latency[j], 0.95))
                    .put("ordered_top3_matches", matches[j]);
        }
        report.put("finish_unix_ms", System.currentTimeMillis()).put("models", stats).put("rows", observations);
        String filename = stages ? "stage-profile.json" : "benchmark.json";
        File temp = new File(app.getFilesDir(), filename + ".tmp");
        Files.write(temp.toPath(), report.toString(2).getBytes(StandardCharsets.UTF_8));
        Files.move(temp.toPath(), new File(app.getFilesDir(), filename).toPath(),
                StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        return report;
    }
}
