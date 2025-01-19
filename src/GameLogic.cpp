#include "GameLogic.hpp"
#include "Video.hpp"
#include <iostream>
#include <cmath>

GameLogic::GameLogic(TrackingPipeline &trackingPipeline)
    : tracking(trackingPipeline), score(0) {}

GameLogic::~GameLogic() {}

void GameLogic::runSingleMode(cv::VideoCapture &cap)
{
    cv::Mat frame;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "Video-Tracking abgeschlossen." << std::endl;
            break;
        }

        try
        {
            // Verarbeite das Frame mit der Tracking-Pipeline
            tracking.processFrame(frame);
            const std::vector<cv::Point> &contour = tracking.getTrackedContour();

            // Füge Bälle hinzu und zeichne sie
            frame = addBallsToPicture(frame);

            // Zeichne die Kontur der verfolgten Person
            if (!contour.empty())
            {
                std::vector<std::vector<cv::Point>> contoursToDraw = {contour};
                cv::drawContours(frame, contoursToDraw, -1, cv::Scalar(0, 255, 0), 2); // Grün
            }

            // Kollisionen prüfen
            handleCollisions(contour);

            // Punktzahl anzeigen
            drawScore(frame);

            // Ergebnis anzeigen
            cv::imshow("Single Mode", frame);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Fehler beim Tracking: " << e.what() << std::endl;
            break;
        }

        // Beende bei 'q'
        if (cv::waitKey(30) == 'q')
        {
            std::cout << "Spiel manuell beendet." << std::endl;
            break;
        }
    }
    cv::destroyAllWindows(); // Fenster schließen
}

void GameLogic::handleCollisions(const std::vector<cv::Point> &contour)
{
    ballList.remove_if([&contour, this](Ball &ball)
                       {
        for (const cv::Point& point : contour) {
            if (cv::norm(cv::Point(ball.getX(), ball.getY()) - point) <= ball.radius) {
                // Punktzahl erhöhen und Ball entfernen
                score++;
                std::cout << "Punkt erzielt! Aktuelle Punktzahl: " << score << std::endl;
                return true;
            }
        }
        return false; });
}

void GameLogic::drawScore(cv::Mat &frame)
{
    cv::putText(frame, "Punktzahl: " + std::to_string(score), cv::Point(10, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
}
