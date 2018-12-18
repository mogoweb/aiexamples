package com.mogoweb.thuglife;

import android.support.annotation.NonNull;

import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.common.FirebaseVisionImageMetadata;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;
import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.Frame;
import com.otaliastudios.cameraview.FrameProcessor;

import java.util.List;

/**
 *
 * FaceProcessor takes the camera frames from CameraView and uses FirebaseVisionFaceDetector
 * to detect the face, and then pass the detected face info to OverlayView so it can draw bitmaps on the face
 *
 * Created by Qichuan on 21/6/18.
 */
class FaceProcessor {
    private CameraView cameraView;
    private OverlayView overlayView;

    public FaceProcessor(CameraView cameraView, OverlayView overlayView) {
        this.cameraView = cameraView;
        this.overlayView = overlayView;
    }

    // Initialize the face detection option, and we need all the face landmarks
    private FirebaseVisionFaceDetectorOptions options = new FirebaseVisionFaceDetectorOptions.Builder()
            .setLandmarkType(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
            .build();

    // Obtain the FirebaseVisionFaceDetector instance
    private FirebaseVisionFaceDetector detector = FirebaseVision.getInstance().getVisionFaceDetector(options);

    public void startProcessing() {

        // Getting frames from camera view
        cameraView.addFrameProcessor(new FrameProcessor() {
            public void process(@NonNull Frame frame) {
                if (frame.getSize() != null) {
                    int rotation = frame.getRotation() / 90;
                    if (rotation / 2 == 0) {
                        overlayView.previewWidth = cameraView.getPreviewSize().getWidth();
                        overlayView.previewHeight = cameraView.getPreviewSize().getHeight();
                    } else {
                        overlayView.previewWidth = cameraView.getPreviewSize().getHeight();
                        overlayView.previewHeight = cameraView.getPreviewSize().getWidth();
                    }
                    // Build a image meta data object
                    FirebaseVisionImageMetadata metadata = new FirebaseVisionImageMetadata.Builder()
                            .setFormat(FirebaseVisionImageMetadata.IMAGE_FORMAT_NV21)
                            .setWidth(frame.getSize().getWidth())
                            .setHeight(frame.getSize().getHeight())
                            .setRotation(rotation)
                            .build();
                    // Create vision image object, and it will be consumed by FirebaseVisionFaceDetector
                    // for face detection
                    FirebaseVisionImage firebaseVisionImage = FirebaseVisionImage.fromByteArray(frame.getData(), metadata);

                    // Perform face detection
                    detector.detectInImage(firebaseVisionImage).addOnSuccessListener(new OnSuccessListener<List<FirebaseVisionFace>>() {
                        @Override
                        public void onSuccess(List<FirebaseVisionFace> firebaseVisionFaces) {
                            if (firebaseVisionFaces.size() > 0) {
                                // We just need the first face
                                FirebaseVisionFace face = firebaseVisionFaces.get(0);

                                // Pass the face to OverlayView
                                overlayView.face = face;
                            }
                        }
                    });
                }
            }
        });
    }
}

