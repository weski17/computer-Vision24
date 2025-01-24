#include "BackgroundSubtractionPipeline.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/bgsegm.hpp>
#include <iomanip>
#include <filesystem>


// Anonymer Namespace mit Hilfsfunktion für Konsolenausgaben.
namespace {
   void logMessage(const std::string& message) {
       std::cout << message << std::endl;
   }
}
// Konstruktor: Initialisiert die Hintergrundsubtraktoren und Variablen
BackgroundSubtractionPipeline::BackgroundSubtractionPipeline() {
   initializeBackgroundSubtractor();
}


// Methode zur Initialisierung der Hintergrundsubtraktoren
void BackgroundSubtractionPipeline::initializeBackgroundSubtractor() {

    mog2 = cv::createBackgroundSubtractorMOG2();
    mog2->setDetectShadows(true);  
    mog2->setShadowValue(127);     // Schatten-Pixelwert auf 127 setzen
    mog2->setShadowThreshold(0.5); 
    mog2->setVarThreshold(25);
    
    knn = cv::createBackgroundSubtractorKNN();

    knn->setDetectShadows(true);
    knn->setHistory(50);   
    knn->setShadowThreshold(0.5); // Schattenempfindlichkeit
    knn->setDist2Threshold(400.0); // Schwelle zur Unterscheidung naher Objekte
    frameCount = 0;
    minInterval = cv::Mat();
    maxInterval = cv::Mat();

  




}


// Methode zur Bildvorverarbeitung
cv::Mat BackgroundSubtractionPipeline::applyPreprocessing(const cv::Mat& frame) {
   cv::Mat processedFrame = frame.clone();
   cv::Mat grayFrame;


   // Umwandlung in Graustufen
   cv::cvtColor(processedFrame, grayFrame, cv::COLOR_BGR2GRAY);
   // Weichzeichnen zur Rauschunterdrückung
   cv::GaussianBlur(processedFrame, processedFrame, cv::Size(3, 3), 0.2);
   return processedFrame;
}


// MOG2-Hintergrundsubtraktion anwenden
cv::Mat BackgroundSubtractionPipeline::applyMixtureOfGaussians(const cv::Mat& frame) {
   cv::Mat fgMask;
   mog2->apply(frame, fgMask);  // MOG2 auf Frame anwenden
   // Schatten entfernen: Behalte nur Vordergrundpixel (255)
   cv::threshold(fgMask, fgMask, 127, 255, cv::THRESH_BINARY);
   return fgMask;
}


// KNN-Hintergrundsubtraktion anwenden
cv::Mat BackgroundSubtractionPipeline::applyKNN(const cv::Mat& frame) {
   cv::Mat fgMask, fgMaskWithoutShadows, edges, combinedMask;


   // KNN auf Frame anwenden
   knn->apply(frame, fgMask);


   // Schatten entfernen (Pixelwerte unter 200 werden ignoriert)
   cv::threshold(fgMask, fgMaskWithoutShadows, 200, 255, cv::THRESH_BINARY);




   // Kantenextraktion mit Canny, begrenzt auf die FG-Maske
   cv::Mat gray;
   cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);  // Graubild erstellen
   cv::Canny(gray, edges, 50, 150);               // Canny-Kanten erkennen
   cv::bitwise_and(edges, fgMaskWithoutShadows, edges); // Kanten innerhalb der Maske extrahieren


   // Kombination der FG-Maske mit den Kanten
   cv::bitwise_or(fgMaskWithoutShadows, edges, combinedMask);


   return combinedMask;
}


// Methode zur Verbesserung der Maske
cv::Mat BackgroundSubtractionPipeline::improveMask(const cv::Mat& inputMask) {
   cv::Mat grayMask, improvedMask;
   // Falls die Eingabemaske 3-Kanal hat, in Graustufen umwandeln
   if (inputMask.channels() == 3) {
       cv::cvtColor(inputMask, grayMask, cv::COLOR_BGR2GRAY);
   } else {
       grayMask = inputMask.clone();
   }


   // Binarisierung der Maske
   cv::threshold(grayMask, improvedMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  
   // Median-Filter zur Glättung und Rauschreduktion
   cv::medianBlur(improvedMask, improvedMask, 5);
  
   // Strukturierungselement für Morphologie-Operationen
   cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
   // Dilation zur Erkennung von Objektkanten
   cv::dilate(improvedMask, improvedMask, element, cv::Point(-1, -1), 1);
   // Anwendung von Close- und Open-Operationen zur weiteren Glättung
   cv::morphologyEx(improvedMask, improvedMask, cv::MORPH_CLOSE, element);
   cv::morphologyEx(improvedMask, improvedMask, cv::MORPH_OPEN, element);


   return improvedMask;
}


// Methode zur Erstellung einer benutzerdefinierten Hintergrundsubtraktionsmaske
cv::Mat BackgroundSubtractionPipeline::IntervalBGSubtraction(const cv::Mat& frame) {
   if (frame.channels() != 3) {
       std::cerr << "Error: Expected a 3-channel frame." << std::endl;
       return cv::Mat();
   }


   // Hintergrundintervalle initialisieren, falls noch nicht geschehen
   if (frameCount == 0 || minInterval.empty() || maxInterval.empty() || frame.size() != minInterval.size()) {
       minInterval = frame.clone();
       maxInterval = frame.clone();
       frameCount = 0;
   }


   // Aufbau der Intervalle über die ersten 100 Frames
   if (frameCount < 40) {
       for (int y = 0; y < frame.rows; y++) {
           for (int x = 0; x < frame.cols; x++) {
               cv::Vec3b currentPixel = frame.at<cv::Vec3b>(y, x);
               cv::Vec3b& minPixel = minInterval.at<cv::Vec3b>(y, x);
               cv::Vec3b& maxPixel = maxInterval.at<cv::Vec3b>(y, x);


               // Aktualisieren der minimalen und maximalen Pixelwerte
               for (int c = 0; c < 3; c++) {
                   minPixel[c] = std::min(minPixel[c], currentPixel[c]);
                   maxPixel[c] = std::max(maxPixel[c], currentPixel[c]);
               }
           }
       }
       frameCount++;
       return cv::Mat::zeros(frame.size(), CV_8UC1);  // Rückgabe einer leeren Maske bis zum vollständigen Setup
   } else {
       cv::Mat fgMask = cv::Mat::zeros(frame.size(), CV_8UC1);
       cv::Mat shadowMask = cv::Mat::zeros(frame.size(), CV_8UC1);
       int tolerance = std::min(10, frameCount / 10);  // Toleranzwert zur Anpassung
       const float shadowThreshold = 0.5;


       for (int y = 0; y < frame.rows; y++) {
           for (int x = 0; x < frame.cols; x++) {
               cv::Vec3b currentPixel = frame.at<cv::Vec3b>(y, x);
               cv::Vec3b minPixel = minInterval.at<cv::Vec3b>(y, x);
               cv::Vec3b maxPixel = maxInterval.at<cv::Vec3b>(y, x);


               bool isForeground = false;
               bool isShadow = false;
               for (int c = 0; c < 3; c++) {
                   if (currentPixel[c] < minPixel[c] - tolerance || currentPixel[c] > maxPixel[c] + tolerance) {
                       isForeground = true;
                   }
                   if (currentPixel[c] >= minPixel[c] * shadowThreshold && currentPixel[c] < minPixel[c]) {
                       isShadow = true;
                   }
               }


               // Setzen der Pixelwerte in der Maske je nach Hintergrund- oder Schattenkategorie
               if (isForeground && !isShadow) {
                   fgMask.at<uchar>(y, x) = 255;
               } else if (isShadow) {
                   shadowMask.at<uchar>(y, x) = 127;
               }
           }
       }


       fgMask.setTo(0, shadowMask);
       return fgMask;
   }
}


// Lädt den ersten Frame eines Videos als Ground-Truth-Maske
cv::Mat BackgroundSubtractionPipeline::loadFirstFrameAsGroundTruth(const std::string& videoPath) {
   cv::VideoCapture capture(videoPath);
   if (!capture.isOpened()) {
       std::cerr << "Fehler: Video konnte nicht geladen werden!" << std::endl;
       return cv::Mat();
   }


   cv::Mat firstFrame;
   capture >> firstFrame;
   if (firstFrame.empty()) {
       std::cerr << "Fehler: Erster Frame konnte nicht geladen werden!" << std::endl;
       return cv::Mat();
   }


   // Konvertiere in Graustufen und binarisiere für Ground-Truth
   if (firstFrame.channels() == 3) {
       cv::cvtColor(firstFrame, firstFrame, cv::COLOR_BGR2GRAY);
   }
   cv::Mat firstFrameBinary;
   cv::threshold(firstFrame, firstFrameBinary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
   return firstFrameBinary;
}


// Speichert den ersten Frame eines Videos als PNG-Bild
void BackgroundSubtractionPipeline::saveFirstFrameAsPNG(const std::string& videoPath, const std::string& outputPath) {
   cv::VideoCapture cap(videoPath);
   if (!cap.isOpened()) {
       std::cerr << "Fehler beim Öffnen des Videos." << std::endl;
       return;
   }


   cv::Mat firstFrame;
   if (cap.read(firstFrame)) {
       std::filesystem::create_directories(outputPath);  // Sicherstellen, dass der Ausgabeordner existiert
       std::string filename = outputPath + "/first_frame.png";
       if (cv::imwrite(filename, firstFrame)) {
           logMessage("Das erste Frame wurde erfolgreich gespeichert.");
       } else {
           std::cerr << "Fehler beim Speichern des Frames." << std::endl;
       }
   } else {
       std::cerr << "Fehler beim Lesen des ersten Frames." << std::endl;
   }
   cap.release();
}


// Berechnet die Metriken zur Bewertung der Maskengenauigkeit
Metrics BackgroundSubtractionPipeline::calculateMetrics(const cv::Mat& predictedMask, const cv::Mat& groundTruthMask) {
   cv::Mat binarizedPredictedMask, binarizedGroundTruthMask;
   // Binarisierung der Masken, falls erforderlich
   cv::threshold(predictedMask, binarizedPredictedMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
   cv::threshold(groundTruthMask, binarizedGroundTruthMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);


   // Berechnung von True Positives, False Positives, False Negatives und True Negatives
   double tp = cv::countNonZero(binarizedPredictedMask & binarizedGroundTruthMask);
   double fp = cv::countNonZero(binarizedPredictedMask & ~binarizedGroundTruthMask);
   double fn = cv::countNonZero(~binarizedPredictedMask & binarizedGroundTruthMask);
   double tn = cv::countNonZero(~binarizedPredictedMask & ~binarizedGroundTruthMask);


   // Berechnung der Metriken zur Bewertung
   double precision = tp / (tp + fp + 1e-6);
   double recall = tp / (tp + fn + 1e-6);
   double f1 = 2 * (precision * recall) / (precision + recall + 1e-6);
   double accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6);
   double iou = tp / (tp + fp + fn + 1e-6);


   return {precision, recall, f1, accuracy, iou};
}


// Zeigt die berechneten Metriken für verschiedene Hintergrundsubtraktionsmethoden an
void BackgroundSubtractionPipeline::displayMetricsForMethods(const cv::Mat& frame, const cv::Mat& groundTruthMask) {


   cv::Mat knnMask = applyKNN(frame);
   cv::Mat mog2Mask = applyMixtureOfGaussians(frame);
   cv::Mat intervalMask = IntervalBGSubtraction(frame);


   cv::Mat improvedMogMask = improveMask(mog2Mask);
   cv::Mat improvedKnnMask = improveMask(knnMask);
   cv::Mat improvedintervalMask = improveMask(intervalMask);


   Metrics knnMetrics = calculateMetrics(improvedKnnMask, groundTruthMask);
   Metrics mog2Metrics = calculateMetrics(improvedMogMask, groundTruthMask);
   Metrics IntervalMetrics = calculateMetrics(improvedintervalMask, groundTruthMask);


   std::cout << std::fixed << std::setprecision(4);
   std::cout << "KNN - Precision: " << knnMetrics.precision
           << ", Recall: " << knnMetrics.recall
           << ", F1-Score: " << knnMetrics.f1
           << ", Accuracy: " << knnMetrics.accuracy
           << ", IoU: " << knnMetrics.iou << std::endl;


   std::cout << "MOG2 - Precision: " << mog2Metrics.precision
           << ", Recall: " << mog2Metrics.recall
           << ", F1-Score: " << mog2Metrics.f1
           << ", Accuracy: " << mog2Metrics.accuracy
           << ", IoU: " << mog2Metrics.iou << std::endl;


   std::cout << "Interval - Precision: " << IntervalMetrics.precision
           << ", Recall: " << IntervalMetrics.recall
           << ", F1-Score: " << IntervalMetrics.f1
           << ", Accuracy: " << IntervalMetrics.accuracy
           << ", IoU: " << IntervalMetrics.iou << std::endl;
}


// Speichert die Hintergrundsubtraktionsmasken in Videos und zeigt die Metriken an
void BackgroundSubtractionPipeline::saveBackgroundSubtractionResults(const cv::Mat& frame, const cv::Mat& groundTruthMask) {
   cv::Mat pocessedFrame = applyPreprocessing(frame);


   cv::Mat knnMask = applyKNN(pocessedFrame);
   cv::Mat mog2Mask = applyMixtureOfGaussians(pocessedFrame);
   cv::Mat IntervalMask = IntervalBGSubtraction(pocessedFrame);


   cv::Mat improvedMogMask = improveMask(mog2Mask);
   cv::Mat improvedKnnMask = improveMask(knnMask);
   cv::Mat improvedIntervalMask = improveMask(IntervalMask);


   outputVideoMOG2.write(improvedMogMask);  // Speichern in das MOG2-Video
   outputVideoKNN.write(improvedKnnMask);    // Speichern in das KNN-Video
  // outputVideoCodebook.write(improvedIntervalMask); // Speichern in das Codebook-Video
  
   cv::waitKey(1);   //Kurze Wartezeit zur Anzeige
}



void BackgroundSubtractionPipeline::initializeVideoWriters(const std::string &outputDirectory, cv::VideoCapture &cap) {
   int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
   int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
   cv::Size frameSize(frameWidth, frameHeight);


   // Direkte Initialisierung der VideoWriter-Objekte
   outputVideoMOG2 = cv::VideoWriter(outputDirectory + "/MOG2IMPROVED.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, frameSize, false);
   outputVideoKNN = cv::VideoWriter(outputDirectory + "/KNNIMPROVED.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, frameSize, false);
   outputVideoCodebook = cv::VideoWriter(outputDirectory + "/CodebookIMPROVED.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, frameSize, false);
   logMessage("Videos werden erstellt ...");
}



void BackgroundSubtractionPipeline::releaseVideoWriters() {
   outputVideoMOG2.release();
   outputVideoKNN.release();
   outputVideoCodebook.release();
}
bool BackgroundSubtractionPipeline::initializeVideoAndLoadGroundTruth(const std::string& videoPath, cv::VideoCapture& cap, cv::Mat& groundTruthMask, const std::string& outputPath) {
   // Videoquelle öffnen und prüfen
   cap.open(videoPath);
   if (!cap.isOpened()) {
       std::cerr << "Fehler beim Öffnen der Videoquelle!" << std::endl;
       return false;
   }


   // Ground-Truth-Maske laden
   groundTruthMask = loadFirstFrameAsGroundTruth(videoPath);
   if (groundTruthMask.empty()) {
       std::cerr << "Fehler: Referenzbild konnte nicht geladen werden!" << std::endl;
       return false;
   }


   // Ground-Truth-Maske als PNG speichern, falls gewünscht
   saveFirstFrameAsPNG(videoPath, outputPath);


   return true;
}
