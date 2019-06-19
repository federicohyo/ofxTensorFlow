#pragma once

#include "ofMain.h"
#include "ofxDatGui.h"
#include "ofxDVS.hpp"
#include "ofxMSATensorFlow.h"

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		
    
        // Silicon Retina
        ofxDVS dvs;
    
        //Gui
        string getHex(int hex);
        void changeDrawGui();
        ofxDatGuiFolder* f1;
        void onButtonEvent(ofxDatGuiButtonEvent e);
        void onToggleEvent(ofxDatGuiToggleEvent e);
        void onSliderEvent(ofxDatGuiSliderEvent e);
        void onMatrixEvent(ofxDatGuiMatrixEvent e);
        void on2dPadEvent(ofxDatGui2dPadEvent e);
        void onTextInputEvent(ofxDatGuiTextInputEvent e);
        void onColorPickerEvent(ofxDatGuiColorPickerEvent e);
        ofxDatGuiTextInput * myTextTimer;
        ofxDatGuiTextInput * myTempReader;
        ofxDatGuiValuePlotter * myIMU;
        bool drawGui;
    
        //counters
        int numPaused;
        int numPausedRec;
    
    
        // tensorflow stuff
        // classifies pixels
        // check the src of this class (ofxMSATFImageClassifier) to see how to do more generic stuff with ofxMSATensorFlow
        msa::tf::ImageClassifier classifier;
    
        void init_classifier();
        int dim_input_x = 64;
        int dim_input_y = 64;


};
