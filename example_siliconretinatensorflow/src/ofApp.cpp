#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    dvs.setup();
    
    //GUI
    int x = 0;
    int y = 0;
    ofSetWindowPosition(0, 0);
    f1 = new ofxDatGuiFolder("Control", ofColor::fromHex(0xFFD00B));
    f1->addBreak();
    f1->addFRM();
    f1->addBreak();
    f1->addSlider("1/speed", 0, 2, dvs.targetSpeed);
    myTextTimer = f1->addTextInput("TIME", dvs.timeString);
    myTempReader = f1->addTextInput("IMU TEMPERATURE", to_string((int)(dvs.imuTemp)));
    f1->addToggle("APS", true);
    f1->addBreak();
    f1->addToggle("DVS", true);
    f1->addBreak();
    f1->addToggle("IMU", true);
    f1->addBreak();
    f1->addMatrix("DVS Color", 7, true);
    f1->addBreak();
    f1->addButton("Clear");
    f1->addBreak();
    f1->addButton("Pause");
    f1->addBreak();
    f1->addButton("Start Recording");
    f1->addBreak();
    f1->addButton("Load Recording");
    f1->addBreak();
    f1->addButton("Live");
    f1->addBreak();
    f1->addToggle("Draw IMU", false);
    f1->addMatrix("3D Time", 4, true);
    f1->addToggle("Pointer", false);
    f1->addToggle("Raw Spikes", true);
    f1->addToggle("DVS Image Gen", false);
    f1->addSlider("BA Filter dt", 1, 100000, dvs.BAdeltaT);
    f1->addSlider("DVS Integration", 1, 100, dvs.fsint);
    f1->addSlider("DVS Image Gen", 1, 20000, dvs.numSpikes);
    //myIMU = f1->addValuePlotter("IMU", 0, 1);
    f1->setPosition(x, y);
    f1->expand();
    f1->onButtonEvent(this, &ofApp::onButtonEvent);
    f1->onToggleEvent(this, &ofApp::onToggleEvent);
    f1->onSliderEvent(this, &ofApp::onSliderEvent);
    f1->onMatrixEvent(this, &ofApp::onMatrixEvent);
    f1->onTextInputEvent(this, &ofApp::onTextInputEvent);
    
    numPaused = 0;
    numPausedRec = 0;
    
    // alpha blend
    //glEnable(GL_BLEND);
    //glBlendFunc( GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA );

    // init tensorflow classifier
    init_classifier();
}

//--------------------------------------------------------------
void ofApp::update(){
    dvs.update();
    dvs.updateBAFilter();
    dvs.updateImageGenerator();
    
    //GUI
    f1->update();
    myTextTimer->setText(dvs.timeString);
    //float val = ofRandom(0, 1);
    //cout << val << endl;
    //myIMU->setValue(val);
    myTempReader->setText(to_string((int)(dvs.imuTemp)));
    
    //classify on new images from imagegenerator after having resized them
    if(classifier.isReady() && dvs.newImageGen) {
        ofLog(OF_LOG_NOTICE, "RUN CLASSIFIER");

        ofImage toclassify;
        toclassify.allocate(dvs.sizeX, dvs.sizeY, OF_IMAGE_GRAYSCALE);
        int i = 0;
        while ( i < toclassify.getPixels().size() ) {
            toclassify.getPixels()[i] = dvs.imageGenerator.getPixels()[i];
            i++;
        }
        //toclassify.rotate90(2);
        toclassify.update();
        
        toclassify.resize(dim_input_x, dim_input_y);//resize(28, 28);
        toclassify.update();
        
        dvs.newImageGen = false; // can i do this safely? probably yes
        //scale image of classifier to match input dimensions
        // pass the pixels to the image classifier...
        //pixels = toclassify.getPixels();
        classifier.classify(toclassify.getPixels());
        int nclass = classifier.getNumClasses();

        // get probability class
        for(int i=0; i<classifier.getNumClasses(); i++) {
            float p = classifier.getClassProbs()[i];
            ofLog(OF_LOG_NOTICE, "classifier nclass %d prob %f", i, p);
        }
        
    }
    
}

//--------------------------------------------------------------
void ofApp::draw(){
    
    dvs.draw();
    
    //GUI
    if(drawGui){
        f1->draw();
    }

    //overlay classification
    stringstream str_outputs;


    if(classifier.isReady() && classifier.getNumClasses() > 0) {
        float cur_y = 0;
        ofSetColor(255);

        // DRAW LAYER PARAMETERS (only really useful for the single layer version, deeper networks need more complex visualization)
        //cur_y += layer_viz.draw(0, cur_y, ofGetWidth());


        // DRAW OUTPUT PROBABILITY BARS
        float box_spacing = ofGetWidth() / classifier.getNumClasses();
        float box_width = box_spacing * 0.8;

        for(int i=0; i<classifier.getNumClasses(); i++) {
            float p = classifier.getClassProbs()[i]; // probability of this label

            // draw probability bar
            float h = (ofGetHeight() - cur_y) * p;
            float x = ofMap(i, 0, classifier.getNumClasses()-1, 0, ofGetWidth() - box_spacing);
            x += (box_spacing - box_width)/2;

            ofSetColor(ofLerp(50.0, 255.0, p), ofLerp(100.0, 0.0, p), ofLerp(150.0, 0.0, p));
            ofDrawRectangle(x, ofGetHeight(), box_width, -h);

            str_outputs << ofToString(classifier.getClassProbs()[i], 3) << " ";

            // draw text
            ofDrawBitmapString(ofToString(i) + ": " + ofToString(p, 2), x, ofGetHeight() - h - 10);
        }


        // draw line indicating top score
        ofSetColor(200);
        ofDrawLine(0, cur_y, ofGetWidth(), cur_y);

    }
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if (key == 'c') {
        changeDrawGui();
    }
}

void ofApp::changeDrawGui(){
    if(drawGui){
        drawGui = false;
    }else{
        drawGui = true;
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){
    
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){
    
}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){
    
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){
    
}

void ofApp::onButtonEvent(ofxDatGuiButtonEvent e)
{
    if(e.target->getLabel() == "Clear"){
        dvs.clearDraw();
    }else if( (e.target->getLabel() == "Pause") ||  (e.target->getLabel() == "Start")){
        numPaused++;
        if((numPaused % 2) == 0){
            e.target->setLabel("Pause");
        }else{
            e.target->setLabel("Start");
        }
        dvs.changePause();
    }else if( (e.target->getLabel() == "Start Recording") ||  (e.target->getLabel() == "Stop Recording")){
        numPausedRec++;
        if((numPausedRec % 2) == 0){
            e.target->setLabel("Start Recording");
        }else{
            e.target->setLabel("Stop Recording");
        }
        dvs.changeRecordingStatus();
    }else if(e.target->getLabel() == "Load Recording"){
        dvs.loadFile();
    }else if(e.target->getLabel() == "Live"){
        dvs.tryLive();
    }
}

void ofApp::onToggleEvent(ofxDatGuiToggleEvent e)
{
    if(e.target->getLabel() == "APS"){
        dvs.changeAps();
    }else if(e.target->getLabel() == "DVS"){
        dvs.changeDvs();
    }else if(e.target->getLabel() == "IMU"){
        dvs.changeImu();
    }else if (e.target->getLabel() == "DVS Image Gen"){
        dvs.setDrawImageGen(e.target->getChecked());
    }else if(e.target->getLabel() == "Raw Spikes"){
        dvs.setDrawSpikes(e.target->getChecked());
    }else if(e.target->getLabel() == "Pointer"){
        dvs.setPointer(e.target->getChecked());
    }else if(e.target->getLabel() == "Draw IMU"){
        dvs.setDrawImu(e.target->getChecked());
    }
    
}

void ofApp::onSliderEvent(ofxDatGuiSliderEvent e)
{
    if(e.target->getLabel() == "1/speed"){
        cout << "onSliderEvent speed is : " << e.value << endl;
        dvs.setTargetSpeed(e.value);
    }else if(e.target->getLabel() == "DVS Integration"){
        cout << "Integration fsint is : " << e.value << endl;
        dvs.changeFSInt(e.value);
    }else if( e.target->getLabel() == "BA Filter dt"){
        cout << "BackGround Filter dt : " << e.value << endl;
        dvs.changeBAdeltat(e.value);
    }else if( e.target->getLabel() == "DVS Image Gen"){
        cout << "Accumulation value : " << e.value << endl;
        dvs.setImageAccumulatorSpikes(e.value);
    }
    dvs.myCam.reset(); // no mesh turning when using GUI
}

void ofApp::onTextInputEvent(ofxDatGuiTextInputEvent e)
{
    cout << "onTextInputEvent" << endl;
}

void ofApp::on2dPadEvent(ofxDatGui2dPadEvent e)
{
    cout << "on2dPadEvent" << endl;
}

void ofApp::onColorPickerEvent(ofxDatGuiColorPickerEvent e)
{
    cout << "onColorPickerEvent" << endl;
}

void ofApp::onMatrixEvent(ofxDatGuiMatrixEvent e)
{
    
    if( e.target->getLabel() == "3D Time"){
        e.target->setRadioMode(true);
        for(size_t i = 0; i < 4 ; i++){
            if(e.child == i){
                dvs.set3DTime(i);
            }
        }
    }else if(e.target->getLabel() == "DVS Color"){
        e.target->setRadioMode(true);
        for(size_t i = 0; i < 6 ; i++){
            if(e.child == i){
                dvs.changeColor(i);
            }
        }
    }
}

void ofApp::init_classifier(){
    // initialize the image classifier, lots of params to setup
    msa::tf::ImageClassifier::Settings settings;
    
    // these settings are specific to the model
    // settings which are common to both models

    if(false){
        settings.image_dims = { 28, 28, 1 };
        settings.itensor_dims = { 1, 28 * 28 };
        settings.labels_path = "";
        settings.input_layer_name = "x_inputs";
        settings.output_layer_name = "y_outputs";
        settings.dropout_layer_name = "";
        settings.varconst_layer_suffix = "_VARHACK";
        settings.norm_mean = 0;
        settings.norm_stddev = 0;
    
        settings.model_path = "models/mnist-deep.pb";
        settings.dropout_layer_name = "keep_prob";
    }
    if(true){
        settings.image_dims = { dim_input_x, dim_input_y, 1};
        settings.itensor_dims = { 1, dim_input_x , dim_input_y, 1};
        settings.labels_path = "";
        settings.input_layer_name = "input_img_input";
        settings.output_layer_name = "output_prob/Softmax";
        settings.varconst_layer_suffix = "";
        settings.norm_mean = 0.5;
        settings.norm_stddev = 0.2;
        // settings which are specific to the individual models
        settings.model_path = "models/model_fingers_tf_6subj_bk.pb";   // load appropiate model file
        settings.dropout_layer_name = "";    // this model has dropout, so need to set keep probability to 1.0
    }
    // initialize classifier with these settings
    classifier.setup(settings);
    if(!classifier.getGraphDef()) {
        ofLogError() << "Could not initialize session. Did you download the data files and place them in the data folder? ";
        ofLogError() << "Download from https://github.com/memo/ofxMSATensorFlow/releases";
        ofLogError() << "More info at https://github.com/memo/ofxMSATensorFlow/wiki";
        assert(false);
        ofExit(1);
    }
    
}
