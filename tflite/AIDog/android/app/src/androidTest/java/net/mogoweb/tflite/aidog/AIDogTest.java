package net.mogoweb.tflite.aidog;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Environment;
import android.support.test.InstrumentationRegistry;
import android.support.test.runner.AndroidJUnit4;
import android.support.v4.content.ContextCompat;
import android.util.Log;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class AIDogTest {
    private final static String TAG = "AIDogTest";
    @Test
    public void useAppContext() {
        // Context of the app under test.
        Context appContext = InstrumentationRegistry.getTargetContext();

        assertEquals("net.mogoweb.tflite.aidog", appContext.getPackageName());
    }

    @Test
    public void predictImages() {
        List<String> list = new ArrayList<String>();

        boolean hasPermission = true;
        int currentAPIVersion = Build.VERSION.SDK_INT;
        if (currentAPIVersion >= android.os.Build.VERSION_CODES.M) {
            if (ContextCompat.checkSelfPermission(InstrumentationRegistry.getTargetContext(), Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                hasPermission = false;
            }
        }
        if (!hasPermission) {
            assertTrue(false);
            return;
        }

        ImageClassifier classifier = null;
        try {
            classifier = new ImageClassifier(InstrumentationRegistry.getTargetContext());
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize an image classifier.");
        }

        File imagesDirectory = new File(Environment.getExternalStorageDirectory().getPath() + "/Images");
        if (imagesDirectory.isDirectory()) {
            for (File dir : imagesDirectory.listFiles()) {
                if (dir.isDirectory()) {
                    String dirName = dir.getName();
                    String trueLabel = dirName.replace('-', ' ').replace('_', ' ');

                    Log.i(TAG, "trueLabel:" + trueLabel);
                    for (File file : dir.listFiles()) {
                        String path = file.getAbsolutePath();
                        if (path.endsWith(".jpg") || path.endsWith(".jpeg") || path.endsWith(".png")) {
                            list.add(path);
                            Log.i(TAG, path);
                            Bitmap bitmap = BitmapFactory.decodeFile(path);
                            String result = classifier.classifyFrame(bitmap);
                            Log.i(TAG, result);
                        }
                    }
                }
            }
        }

        assertTrue(list.size() > 0);
    }
}
