#ifndef GAMELOGIC_HPP
#define GAMELOGIC_HPP

#include <opencv2/opencv.hpp>
#include "TrackingPipeline.hpp"
#include "Ball.hpp"
#include "BallFunktions.hpp"

class GameLogic
{
public:
    GameLogic(TrackingPipeline &trackingPipeline);
    ~GameLogic();

    void runSingleMode(cv::VideoCapture &cap);

private:
    TrackingPipeline &tracking; // Referenz auf die Tracking-Pipeline
    int score;                  // Punktzähler

    void handleCollisions(const std::vector<cv::Point> &contour);
    void drawScore(cv::Mat &frame);
};

#endif // GAMELOGIC_HPP
