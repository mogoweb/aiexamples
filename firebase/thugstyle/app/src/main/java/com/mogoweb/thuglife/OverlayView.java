package com.mogoweb.thuglife;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.View;

import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceLandmark;

public class OverlayView extends View {
    // The detected face
    FirebaseVisionFace face = null;

    public void setFace(FirebaseVisionFace value) {
        face = value;

        // Trigger redraw when a new detected face object is passed in
        postInvalidate();
    }

    int previewWidth = -1;
    int previewHeight = -1;

    private float widthScaleFactor = 1.0f;
    private float heightScaleFactor = 1.0f;

    // The glasses bitmap
    private Bitmap glassesBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.glasses);

    // The cigarette bitmap
    private Bitmap cigaretteBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.cigarette);

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    @Override
    public void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        // Create local variables here so they canot not be changed anywhere else
        FirebaseVisionFace face = this.face;
        int previewWidth = this.previewWidth;
        int previewHeight = this.previewHeight;

        if (face != null && canvas != null && previewWidth != -1 && previewHeight != -1) {

            // Calculate the scale factor
            widthScaleFactor = canvas.getWidth() / (float)previewWidth;
            heightScaleFactor = canvas.getHeight() / (float)previewHeight;

            drawGlasses(canvas, face);
            drawCigarette(canvas, face);
        }
    }

    /***
     * Draw glasses on top of eyes
     */
    private void drawGlasses(Canvas canvas, FirebaseVisionFace face) {
        FirebaseVisionFaceLandmark leftEye = face.getLandmark(FirebaseVisionFaceLandmark.LEFT_EYE);
        FirebaseVisionFaceLandmark rightEye = face.getLandmark(FirebaseVisionFaceLandmark.RIGHT_EYE);

        if (leftEye != null && rightEye != null) {
            float eyeDistance = leftEye.getPosition().getX() - rightEye.getPosition().getY();
            int delta = (int)(widthScaleFactor * eyeDistance / 2);
            Rect glassesRect = new Rect(
                    (int)translateX(leftEye.getPosition().getX()) - delta,
                    (int)translateY(leftEye.getPosition().getY()) - delta,
                    (int)translateX(rightEye.getPosition().getX()) + delta,
                    (int)translateY(rightEye.getPosition().getY()) + delta);
            canvas.drawBitmap(glassesBitmap, null, glassesRect, null);
        }
    }

    /**
     * Draw cigarette at the left mouth
     */
    private void drawCigarette(Canvas canvas, FirebaseVisionFace face) {
        FirebaseVisionFaceLandmark rightMouth = face.getLandmark(FirebaseVisionFaceLandmark.RIGHT_MOUTH);
        FirebaseVisionFaceLandmark leftMouth = face.getLandmark(FirebaseVisionFaceLandmark.LEFT_MOUTH);

        if (leftMouth != null && rightMouth != null) {
            int mouthLength = (int)((leftMouth.getPosition().getX() - rightMouth.getPosition().getX()) * widthScaleFactor);
            Rect cigaretteRect = new Rect(
                    (int)translateX(leftMouth.getPosition().getX()) - mouthLength,
                    (int)translateY(leftMouth.getPosition().getY()),
                    (int)translateX(leftMouth.getPosition().getX()),
                    (int)translateY(leftMouth.getPosition().getY()) + mouthLength
            );

            canvas.drawBitmap(cigaretteBitmap, null, cigaretteRect, null);
        }
    }

    /**
     * Adjusts the x coordinate from the preview's coordinate system to the view coordinate system.
     */
    private float translateX(float x) {
        return getWidth() - scaleX(x);
    }

    /**
     * Adjusts the y coordinate from the preview's coordinate system to the view coordinate system.
     */
    private float translateY(float y) {
        return scaleY(y);
    }

    /** Adjusts a vertical value of the supplied value from the preview scale to the view scale. */
    private float scaleX(float x) {
        return x * widthScaleFactor;
    }


    /** Adjusts a vertical value of the supplied value from the preview scale to the view scale. */
    private float scaleY(float y) {
        return y * heightScaleFactor;
    }
}
