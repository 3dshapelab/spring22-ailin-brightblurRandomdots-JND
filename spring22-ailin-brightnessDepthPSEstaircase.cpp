// Part 2.1 of the brightness depth experiment: PSE (part 2.2 for JND)
// compare to fall21-brightnessDepth version, the dim dots can be blurred in this version

// Background:
// In Part 1 (fall20-brightnessDepth), we use probe task and found out the brighter the dots are, the deeper the cylinder is perceived to be
// but that can also be explained by the cue-to-flatness. To have dinstinctive predictions between IC and Bayesian, we test the JND

// Design:
// Factorial: bright/dim of standard * bright/dim of comparisons * horizontal * 2 depths
// 2IFC comparison task - std and comp



// 4 sessions in Part 2:      !! THIS CPP runs Session 1 and Session 2!!
// Session 1: 1-up-1-down & 1-down-1-up to get PSE. Standard: Bright; Comparison: Bright & Dim
// Session 2: 1-up-1-down & 1-down-1-up to get PSE. Standard: Dim; Comparison: Bright & Dim

// Session 3 & 4 have different starting values - approaching the thresholds from opposite directions

// Session 3: 3-up-1-down & 1-down-3-up to get JND. 
// One starting value < flater standard depth; the other starting value > deeper standard depth;

// Session 4: 3-up-1-down & 1-down-3-up to get JND.
// flater standard depth < two starting values < deeper standard depth

// Major Differences in PSE.cpp and JND.cpp
// ParametersLoader
// parametersFile_directory
// initTrial(): not reading "stdBrightLvl" from parameter file for PSE.cpp
// initTrial(): depth_std are chosen differently
// advanceTrial(): "percentComplete"


// History
// 3/10/2022 dim dots are also blurred. distractors appear after each trial

// It goes like this:
// starts with a constant stimulus that does not go away (constantPres). can use keypress to change its depth
// pressing '+': 'constantPres' -> 'training', which is 2IFC
// pressing 't': 'training' -> 'experiment' which is the real experiment and we start recording responses on txt file
// every 60 trials it goes into 'breakTime'
// pressing '+': 'breakTime' -> 'experiment'


// What can be changed:
// bool blur_one_eye
// bool resetScreen_betweenRuns
// 3 shapes to choose in the constant present mode

// Procedures:
// Session 1:
// 1. go to "parameters_spring22-ailin-brightnessDepth_staircase_main"
// 2. subjName and IOD, set session to 1
// 3. After session 1, calculate the PSEs 

// Session 2:
// 1. go to "parameters_spring22-ailin-brightnessDepth_staircase_main"
// 2. set session to 2
// 3. put in PSEflat and PSEdeep using results from session 1 

#pragma once
// The following macros define the minimum required platform.  The minimum required platform
// is the earliest version of Windows, Internet Explorer etc. that has the necessary features to run 
// your application.  The macros work by enabling all features available on platform versions up to and 
// including the version specified.

// Modify the following defines if you have to target a platform prior to the ones specified below.
// Refer to MSDN for the latest info on corresponding values for different platforms.
#ifndef _WIN32_WINNT            // Specifies that the minimum required platform is Windows Vista.
#define _WIN32_WINNT 0x0600     // Change this to the appropriate value to target other versions of Windows.
#endif

#include <cstdlib>
#include <cmath>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <queue>
#include <string>


/**** BOOOST MULTITHREADED LIBRARY *********/
#include <boost/thread/thread.hpp>
#include <boost/asio.hpp>	//include asio in order to avoid the "winsock already declared problem"


#ifdef _WIN32
#include <windows.h>
#include "CShaderProgram.h" // shader program

#include <gl\gl.h>            // Header File For The OpenGL32 Library
#include <gl\glu.h>            // Header File For The GLu32 Library
#include "glut.h"            // Header File For The GLu32 Library
#include <MMSystem.h>
#endif

/************ INCLUDE CNCSVISION LIBRARY HEADERS ****************/
#include "Mathcommon.h"
#include "GLUtils.h"
#include "VRCamera.h"
#include "CoordinatesExtractor.h"
#include "CylinderPointsStimulus.h"
#include "EllipsoidPointsStimulus.h"
#include "StimulusDrawer.h"
#include "GLText.h"
#include "TrialGenerator.h"
#include "ParametersLoader.h"
#include "Util.h"
#include "VRCamera.h"
#include "BalanceFactor.h"
#include "ParStaircase.h"
#include "Staircase.h"
#include "BrownPhidgets.h"
#include <direct.h>
#include "Optotrak2.h"
#include "Marker.h"

//Library for texture mapping
#include "SOIL.h"
/***** CALIBRATION FILE *****/
#include "LatestCalibration.h"

/********* NAMESPACE DIRECTIVES ************************/
using namespace std;
using namespace mathcommon;
using namespace Eigen;
using namespace util;


#include <direct.h>
#include "Optotrak2.h"
#include "Marker.h"
#include "BrownMotorFunctions.h"
using namespace BrownMotorFunctions;
using namespace BrownPhidgets;



/*************** Variable Declarations ********************/
static const bool gameMode = true;
const float DEG2RAD = M_PI / 180;
/********* VARIABLES OBJECTS  **********************/
VRCamera cam;
Optotrak2* optotrak;
Screen screen;
CoordinatesExtractor headEyeCoords;
/********** STREAMS **************/
ofstream responseFile;
/********* November 2019   CALIBRATION ON CHIN REST *****/

static const Vector3d center(0, 0, focalDistance);
double mirrorAlignment = 0.0, screenAlignmentY = 0.0, screenAlignmentZ = 0.0;

/********** EYES AND MARKERS **********************/
Vector3d eyeLeft, eyeRight, eyeMiddle;
vector <Marker> markers;
double interoculardistance = 60.0;
int screen1 = 19, screen2 = 20, screen3 = 21;
int mirror1 = 6, mirror2 = 22;

/********** TRIAL SPECIFIC PARAMETERS ***************/
ParametersLoader parameters;
TrialGenerator<double> trial;

int sessionNum = 1; //it is determined by the sessionNum on the parameter file
int blkNum = 1;
int trialNum = 0;
int stairID = 0, stair_reversal = 0, ascending = 0;
double percentComplete = 0;

//controling bool
bool training = true;
bool visibleInfo = true;
bool finished = false;
bool dots_built = false;
bool stdFirst = true;
bool resp_firstDeeper = false;
bool resp_compDeeper = true;
bool is_distractor = false;

// Time variables 
Timer trial_time;
enum Stages { constantPresent, preparation, firstFixation, firstPresent, interStimBreak, secondPresent, respond, distractorPresent, waitForNextTrial, breakTime, expFinished };
Stages currentStage = constantPresent; // if just want to look at the stimuli, use the constant present stage
double ElapsedTime = 0;
double presentTime = 0;
double respondTime = 0;
double presentationTime = 600, fixationTime = 500;
double distractorTime = 400, blankTime_beforeDistractor = 300, blankTime_afterDistractor = 1500;

/********** STIMULUS SPECIFIC PARAMETERS ***************/
// the stimulus
int cylHorizontal = 1; //used to call rotation_bool in brightnessDepth.cpp
double display_distance = -400;
double distDist_jitter_first = 0, distDist_jitter_second = 0;


double visual_angle = 7; // stim diangonal size
double stimulus_height = 70;
double stimulus_width = 1.1 * stimulus_height;
double Gaussian_sig_height_ratio = 0.14;
double Gaussian_sig = Gaussian_sig_height_ratio * stimulus_height;

double depth_constPres = 40; // for constant present
int depthLevel_std = 0;
double flatStdDepth_dimStd = 30, deepStdDepth_dimStd = 45;
double depth_std = 30, depth_comp = 10, depth_first = 30, depth_second = 10;
double depth_distractor = 30;

// dot specs
int dot_per_col = 16, dot_per_row = 16; //16, 19
int dot_number = (dot_per_col * dot_per_row) /(1.5 * 1.5);//1.5 is the ration of temp_height / stimulus_height
double dot_visangle = .1333;
double jitter_x_max = 2.5, jitter_y_max = 2.5;
std::vector<Vector3d> dotContainer_first;
std::vector<Vector3d> dotContainer_second;
std::vector<Vector3d> dotContainer_distractor;

// brightness
int brightnessLevel_std = 1, brightnessLevel_comp = 1;
int brightnessLevel_first = 0, brightnessLevel_second = 0;
float brightness_constPres = 1.0;
float brightness_first = 0.8, brightness_second = 0.8, brightness_std = 0.8, brightness_comp = 0.8;


/********** MODE OF EXPERIMENT ***************/
bool resetScreen_betweenRuns = false; // do we want to reset the screen between blocks

// stimulus control
enum ShapeTypes {cylinder, gaussian, cosine};
ShapeTypes expShape = cylinder;

int panelState = 1; // 0 - no aperture; 1 - aperture black; 2 - aperture background color
bool blur_left_eye = true;
bool blur_one_eye = true;

int blurringBox_Width = 3;
int blur_extraPasses = 3;
double dot_size_bfBlur = 1.14; // when extrapass is 4 -> 1.24
double dot_radius_blur = atan(dot_size_bfBlur / abs(display_distance - 30));


/*************************** INPUT AND OUTPUT ****************************/
string subjectName;

// experiment directory
string experiment_directory = "C:/Users/visionlab/Documents/data/ailin/spring22-ailin-brightnessDepthJNDstaircase/";

// paramters file directory and name
string parametersFileName = experiment_directory + "/parameters_spring22-ailin-brightnessDepth_staircase_main.txt";

// response file headers
string responseFile_headers = "subjName\tIOD\tblockN\ttrialN\tdisplayDistance\tvisualAngle\tdotNum\tclyHorizontal\tstdBrightness\tcompBrightness\tstdDepth\tcompDepth\tresp_firstDeeper\tresp_compDeeper\tID\treversals\tascending\tRT";


/*************************** Blur Shader ********************************************/
// important, don't forget that you include the CShaderProgram.h at the very top of the experiment file
int Width = SCREEN_WIDTH; int Height = SCREEN_HEIGHT;
GLuint sceneBuffer;
GLuint FBO;
CShaderProgram BlurH, BlurV;
float *data;


/*************************** FUNCTIONS ***********************************/
void initOptotrak();
void initMotors();
void initRendering();
void initVariables();
void initStreams();
void handleResize(int w, int h);
void drawStimulus_clear_eye();
void drawStimulus_blur_eye();
void initTrial();
void initBlock();
void advanceTrial();
void updateTheMarkers();
void drawInfo();
void beepOk(int tone);
void cleanup();
void initProjectionScreen(double _focalDist, const Affine3d& _transformation = Affine3d::Identity(), bool synchronous = true);
void online_apparatus_alignment();
void drawBlockingPanels(double dispDistJitter);
std::vector<Vector3d> buildStereoCylinder(double randomDotsDepth, double dispDistJitter, bool isDistractor);
//std::vector<Vector3d> buildStereoCylinderOld(double dispDepth);
std::vector<Vector3d> buildStereoGaussian(double randomDotsDepth, double dispDistJitter, bool isDistractor);
std::vector<Vector3d> buildStereoCosine(double randomDotsDepth, double dispDistJitter, bool isDistractor);
void drawRandomDots(float dotBrightness, double randomDotsDepth, double dispDistJitter, std::vector<Vector3d> dot_container, bool isDistractor);
void drawFixation(double dispDistJitter);
void online_trial();
void drawProgressBar();

// blur stuff
void initBlur();
void inputRandomDots_forBlur(double randomDotsDepth, double dispDistJitter, std::vector<Vector3d> dot_container, bool isDistractor);
void drawRandomDots_blurred(double randomDotsDepth, double dispDistJitter, std::vector<Vector3d> dot_container, bool isDistractor); // to test the blur implementation
void blurPass(CShaderProgram blur_shader, unsigned int texID); // does one pass through using blur with a frame buffer
void drawBlurInput();

/*
// test original stimuli in the probe test 'fall20-ailin-brightnessDepth'
std::vector<Vector3d> buildStereoCylinderOld(double dispDepth)
{
	std::vector<Vector3d> dot_container;
	
	double edge = tan((DEG2RAD * visual_angle)/2) * 2 * (abs(display_distance) + dispDepth) ; 
	double this_cylinder_width = 1.2 * edge;

	//stimulus_height = 55;
	for (int dots_placed = 0; dots_placed < 550; dots_placed++)
	{
		double x_axis = rand() % int(this_cylinder_width) - this_cylinder_width/2;
		double y_axis = rand() % int(edge) - edge/2;

		double z_axis = dispDepth * sqrt(1 - pow(y_axis/(edge/2.),2));
	
		dot_container.push_back(Vector3d(x_axis,y_axis,z_axis));


	}
	
	return dot_container;
}

*/

void initBlur()
{
	// Load in the vertex and frament shader files to create two shader programs, one for horz blur and one for vert blur
	BlurH.Load((char*)(experiment_directory + "Shaders/blur.vs").c_str(), (char*)(experiment_directory + "/Shaders/blurh.fs").c_str());
	BlurV.Load((char*)(experiment_directory + "Shaders/blur.vs").c_str(), (char*)(experiment_directory + "/Shaders/blurv.fs").c_str());
	
	// set up uniforms for the horz and vert blur shaders
	BlurH.UniformLocations = new GLuint[2];
    BlurH.UniformLocations[0] = glGetUniformLocation(BlurH, "Width");
    BlurH.UniformLocations[1] = glGetUniformLocation(BlurH, "odw");

    BlurV.UniformLocations = new GLuint[2];
    BlurV.UniformLocations[0] = glGetUniformLocation(BlurV, "Width");
    BlurV.UniformLocations[1] = glGetUniformLocation(BlurV, "odh");

    glUseProgram(BlurH);
    glUniform1i(BlurH.UniformLocations[0], blurringBox_Width);
    glUseProgram(BlurV);
    glUniform1i(BlurV.UniformLocations[0], blurringBox_Width);
    glUseProgram(0);
	

	// not sure yet
	//glGenTextures(3, &sceneBuffer);
	glGenTextures(1, &sceneBuffer);
    glGenFramebuffers(1, &FBO);
    data = new float[4096];
}

// adapted from the drawBlur() in other scripts
void drawRandomDots_blurred(double randomDotsDepth, double dispDistJitter, std::vector<Vector3d> dot_container, bool isDistractor)
{
	// 1. draw stimulus and aperture --------------------------------------------- // 
	glLoadIdentity();
	glTranslated(0, 0, display_distance);
	
	// generate the fbo
	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	
	//bind the fbo
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);  

	//create a texture
	unsigned int firstPass;
	glGenTextures(1, &firstPass);
	glBindTexture(GL_TEXTURE_2D, firstPass);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  

	// attached texture to the framebuffer
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, firstPass, 0);

	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE){

		////////////////////////////////// draw what you want to blur /////////////////
		inputRandomDots_forBlur(randomDotsDepth, dispDistJitter, dot_container, isDistractor);
        ///////////////////////////////////////////////////////////////////////////////
		glBindFramebuffer(GL_FRAMEBUFFER, 0);   
		glDeleteFramebuffers(1, &fbo);  
	} else {
		cout<<"Framebuffer not complete"<<endl;
	}
	///// we have an texture image of the scene called texture

	// 2. apply the (first) horizontal blur --------------------------------------------- // 
	blurPass(BlurH, firstPass);

	// 3. repeat vertical+horizontal blurs to increase the blur magnitude -------------- // 

	//int blur_mag = 1;
	if(blur_extraPasses > 0){
		for(int passes = 0; passes < blur_extraPasses; passes++){
			blurPass(BlurV, firstPass);
			blurPass(BlurH, firstPass);
		}
	}

	// 4. apply (final) vertical blur and draw ----------------------------------------- // 
	// this time, we actually draw the observable scene, instead of drawing it into frame buffer
	// apply vert blur pass and draw

	glUseProgram(BlurV);
	double ds = 1.0;
	glUniform1f(BlurV.UniformLocations[1], 1.0f / (float)(Width / ds));
	
	// Draw final plane with blurred image
	glLoadIdentity();
	glTranslated(0, 0, display_distance);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, firstPass); // is this the 2 buffer?

	//double horzOffset = 1.2 * tan((DEG2RAD * visual_angle)/2) * 2 * (abs(display_distance))/2.0; // ~30
	//double vertOffset = 2.4 * tan((DEG2RAD * visual_angle)/2) * 2 * (abs(display_distance))/3.0;
	double horzOffset = 0;
	double vertOffset = 0;
	
	glBegin(GL_QUADS);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(-SCREEN_WIDE_SIZE/2 - horzOffset, SCREEN_HIGH_SIZE/2 - vertOffset, 0.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(SCREEN_WIDE_SIZE/2 - horzOffset, SCREEN_HIGH_SIZE/2 - vertOffset, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(SCREEN_WIDE_SIZE/2 - horzOffset, -SCREEN_HIGH_SIZE/2 - vertOffset, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(-SCREEN_WIDE_SIZE/2 - horzOffset, -SCREEN_HIGH_SIZE/2 - vertOffset, 0.0f);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
	glDeleteTextures(1, &firstPass);
	glUseProgram(0);

}


void blurPass(CShaderProgram blur_shader, unsigned int texID)
{
	// generate the fbo
	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	
	//bind the fbo
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);  

	// attached texture to the framebuffer
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texID, 0);
	
	glUseProgram(blur_shader);
	float ds = 1.0;

	glUniform1f(blur_shader.UniformLocations[1], 1.0f / (float)(Width / ds));

	// Draw plane with blurred image
	glLoadIdentity();
	glTranslated(0, 0, display_distance);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texID); // is this the 2 buffer?

	//double horzOffset = 1.2 * tan((DEG2RAD * visual_angle)/2) * 2 * (abs(display_distance))/2.0;
	//double vertOffset = 2.4 * tan((DEG2RAD * visual_angle)/2) * 2 * (abs(display_distance))/3.0;
	double horzOffset = 0;
	double vertOffset = 0;

	//draw the plane with the texture saved by the frame buffer on it
	
	glBegin(GL_QUADS);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(-SCREEN_WIDE_SIZE/2 - horzOffset, SCREEN_HIGH_SIZE/2 - vertOffset, 0.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(SCREEN_WIDE_SIZE/2 - horzOffset, SCREEN_HIGH_SIZE/2 - vertOffset, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(SCREEN_WIDE_SIZE/2 - horzOffset, -SCREEN_HIGH_SIZE/2 - vertOffset, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(-SCREEN_WIDE_SIZE/2 - horzOffset, -SCREEN_HIGH_SIZE/2 - vertOffset, 0.0f);
	glEnd();
	


	// unbind texture and shader program after drawing
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
	glUseProgram(0);
	
	// unbind and delete the framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);   
	glDeleteFramebuffers(1, &fbo); 

}


// This function seems to be used to shut down the system after use
void shutdown() {

	responseFile.close(); // close this object
	if(resetScreen_betweenRuns)
		homeEverything(5000, 4500);
	cleanup();
	exit(0);
}
void cleanup()
{
	// Stop the optotrak
	optotrak->stopCollection();
	delete optotrak;
}
void initProjectionScreen(double _focalDist, const Affine3d& _transformation, bool synchronous)
{
	focalDistance = _focalDist;
	screen.setWidthHeight(SCREEN_WIDE_SIZE, SCREEN_WIDE_SIZE * SCREEN_HEIGHT / SCREEN_WIDTH);
	screen.setOffset(alignmentX, alignmentY);
	screen.setFocalDistance(_focalDist);
	screen.transform(_transformation);
	cam.init(screen);
	if (synchronous)
		moveScreenAbsolute(_focalDist, homeFocalDistance, 4500);
	else
		moveScreenAbsoluteAsynchronous(_focalDist, homeFocalDistance, 4500);
}
// Initialize Optotrak for use in the experiment
void initOptotrak()
{
	optotrak = new Optotrak2(); //intiailize the Optotrak object
	optotrak->setTranslation(calibration);

	//define Optotrak-specific variables
	int numMarkers = 22;
	float frameRate = 85.0f;
	float markerFreq = 4600.0f;
	float dutyCycle = 0.4f;
	float voltage = 7.0f;

	// run the intiailization method for the Optotrak, checking to see if ever (if == 0) and catch the error if so
	if (optotrak->init("C:/cncsvisiondata/camerafiles/Aligned20111014", numMarkers, frameRate, markerFreq, dutyCycle, voltage) != 0)
	{
		cerr << "Something during Optotrak initialization failed, press ENTER to continue. A error log has been generated, look \"opto.err\" in this folder" << endl;
		cin.ignore(1E6, '\n');
		exit(0);
	}

	// Read 10 frames of coordinates and fill the markers vector
	for (int i = 0; i < 10; i++)
	{
		updateTheMarkers();
	}
}
// run a method to define a vector that holds marker positions 
void updateTheMarkers()
{
	optotrak->updateMarkers();
	markers = optotrak->getAllMarkers();

}
// Initialize motors for moving screen around
void initMotors()
{
	//specify the speed for (objects,screen)
	if(resetScreen_betweenRuns)
		homeEverything(5000, 4500);
}

// Method that initializes the openGL parameters needed for creating the stimuli. 
// seems like this is not changed for each experiment (maybe for different experimental setup eg monitor)
void initRendering()
{
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	/* Set depth buffer clear value */
	glClearDepth(1.0);
	/* Enable depth test */
	glEnable(GL_DEPTH_TEST);
	/* Set depth function */
	glDepthFunc(GL_LEQUAL);
	// scommenta solo se vuoi attivare lo shading degli stimoli

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	// Tieni questa riga per evitare che con l'antialiasing attivo le linee siano piu' sottili di un pixel e quindi
	// ballerine (le vedi vibrare)
	glLineWidth(1.5);

}


void initStreams()
{

	ifstream parametersFile;
	parametersFile.open(parametersFileName.c_str());
	parameters.loadParameterFile(parametersFile);

	// Subject name
	subjectName = parameters.find("SubjectName");
	// session 
	sessionNum = str2num<int>(parameters.find("session"));

	if(sessionNum > 2){
		string error_on_file_io = string(" wrong cpp");
		cerr << error_on_file_io << endl;
		MessageBox(NULL, (LPCSTR)"WRONG CPP FILE\n Please run the brightnessDepthJND.", NULL, NULL);

		exit(0);
	}

	// trialFile directory
	string dirName = experiment_directory + subjectName;
	mkdir(dirName.c_str()); // windows syntax

	// Throw an error if the subject name is already used. Don't want to overwrite good data!
	if (util::fileExists(dirName + "/" + subjectName + "_s" + parameters.find("session") + ".txt")) // when writing distractor task data to file, use this style of filename
	{
		string error_on_file_io = subjectName + string(" already exists");
		cerr << error_on_file_io << endl;
		MessageBox(NULL, (LPCSTR)"FILE ALREADY EXISTS\n Please check the parameters file.", NULL, NULL);
		exit(0);
	}
	string responseFileName = dirName + "/" + subjectName + "_s" + parameters.find("session") + ".txt";

	interoculardistance = atof(parameters.find("IOD").c_str());
	display_distance = atof(parameters.find("dispDistance").c_str());
	flatStdDepth_dimStd = atof(parameters.find("PSEflat").c_str());
	deepStdDepth_dimStd = atof(parameters.find("PSEdeep").c_str());

	responseFile.open(responseFileName.c_str());
	responseFile << fixed << responseFile_headers << endl;

}

void drawProgressBar() {
	
	glLoadIdentity();
	glTranslated(0, -55, display_distance);

	glColor3f(0.2, 0.2, 0.2);
	glBegin(GL_LINE_LOOP);
	glVertex3f(-50, 5, 0);
	glVertex3f(50, 5, 0);
	glVertex3f(50, -5, 0);
	glVertex3f(-50, -5, 0);
	glEnd();

	glColor3f(0.1, 0.3, 0.1);
	glBegin(GL_POLYGON);
	glVertex3f(-50, 5, 0);
	glVertex3f(-50 + percentComplete, 5, 0);
	glVertex3f(-50 + percentComplete, -5, 0);
	glVertex3f(-50, -5, 0);
	glEnd();
}

void initBlock()
{

	trial.init(parameters);

	if (sessionNum < 2) {
		brightnessLevel_std = 1;
	}
	else {
		brightnessLevel_std = 0;
	}

	trialNum = 1;
	trial.next(false);
}

void initVariables()
{
	blkNum = (sessionNum - 1) * 2 + 1;
	initBlock();
	// eye coordinates
	eyeRight = Vector3d(interoculardistance / 2, 0, 0);
	eyeLeft = Vector3d(-interoculardistance / 2, 0, 0);
	eyeMiddle = Vector3d(0, 0, 0);
		
	initTrial();
	initProjectionScreen(display_distance);

}

//Central function for projecting image onto the screen
void drawGLScene()
{
	// Draw left eye view
	glDrawBuffer(GL_BACK_RIGHT);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	cam.setEye(eyeLeft);

	if(blur_one_eye){
		drawStimulus_blur_eye();
	}else{
		drawStimulus_clear_eye();
	}

	drawInfo();


	// Draw right eye view
	glDrawBuffer(GL_BACK_LEFT);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	cam.setEye(eyeRight);
	drawStimulus_clear_eye();
	
	drawInfo();
	
	glutSwapBuffers();
}


void update(int value)
{
	glutPostRedisplay();
	glutTimerFunc(TIMER_MS, update, 0);
}

void drawInfo()
{
	// displays relevant information to the screen
	if (visibleInfo)
	{
		glDisable(GL_COLOR_MATERIAL);
		glDisable(GL_BLEND);

		GLText text;

		if (gameMode)
			text.init(SCREEN_WIDTH, SCREEN_HEIGHT, glWhite, GLUT_BITMAP_HELVETICA_18);
		else
			text.init(640, 480, glWhite, GLUT_BITMAP_HELVETICA_12);

		text.enterTextInputMode();

		if (currentStage == expFinished) {

			glColor3fv(glWhite);
			text.draw("The experiment is over. Thank you! :)");

		}
		else if (currentStage == breakTime) {
			glColor3fv(glRed);
			text.draw("Break time! Press + to continue");
		}
		else if (currentStage == constantPresent) {

			glColor3fv(glWhite);
			text.draw("Welcome! press + to start training");
			glColor3fv(glRed);
			text.draw("or press m n a s to change the stimulus");
			text.draw(" ");
			text.draw("# IOD: " + stringify<double>(interoculardistance));
			text.draw("# depth: " + stringify<double>(depth_constPres));

			// check if mirror is calibrated
			if (abs(mirrorAlignment - 45.0) < 0.2)
				glColor3fv(glGreen);
			else
				glColor3fv(glRed);
			text.draw("# Mirror Alignment = " + stringify<double>(mirrorAlignment));

			// check if monitor is calibrated
			if (screenAlignmentY < 89.0)
				glColor3fv(glRed);
			else
				glColor3fv(glGreen);
			text.draw("# Screen Alignment Y = " + stringify<double>(screenAlignmentY));
			if (abs(screenAlignmentZ) < 89.0)
				glColor3fv(glRed);
			else
				glColor3fv(glGreen);
			text.draw("# Screen Alignment Z = " + stringify<double>(screenAlignmentZ));

			
		}
		else {
			//text.draw("# Name: " + subjectName);		
			text.draw("# IOD: " + stringify<double>(interoculardistance));
			text.draw(" ");

			// check if mirror is calibrated
			if (abs(mirrorAlignment - 45.0) < 0.2)
				glColor3fv(glGreen);
			else
				glColor3fv(glRed);
			text.draw("# Mirror Alignment = " + stringify<double>(mirrorAlignment));

			// check if monitor is calibrated
			if (screenAlignmentY < 89.0)
				glColor3fv(glRed);
			else
				glColor3fv(glGreen);
			text.draw("# Screen Alignment Y = " + stringify<double>(screenAlignmentY));
			if (abs(screenAlignmentZ) < 89.0)
				glColor3fv(glRed);
			else
				glColor3fv(glGreen);
			text.draw("# Screen Alignment Z = " + stringify<double>(screenAlignmentZ));
			glColor3fv(glWhite);

			text.draw(" "); text.draw(" "); text.draw(" ");


			glColor3fv(glWhite);
			text.draw("# trial: " + stringify<int>(trialNum));
			
			text.draw("# std depth: " + stringify<double>(depth_std));
			text.draw("# comp depth: " + stringify<double>(depth_comp));
			text.draw("# current stage: " + stringify<int>(currentStage));
			text.draw("# time: " + stringify<double>(ElapsedTime));
		}
		text.leaveTextInputMode();
		glEnable(GL_COLOR_MATERIAL);
		glEnable(GL_BLEND);
	}
}

/***** SOUNDS *****/
void beepOk(int tone)
{

	switch (tone)
	{
	case 0:
		PlaySound((LPCSTR) "C:\\cygwin\\home\\visionlab\\workspace\\cncsvision\\data\\beep\\beep-440-pluck.wav", NULL, SND_FILENAME | SND_ASYNC);
		break;
	case 1:
		PlaySound((LPCSTR) "C:\\cygwin\\home\\visionlab\\workspace\\cncsvision\\data\\beep\\beep-success.wav", NULL, SND_FILENAME | SND_ASYNC);
		break;
	case 3:
		PlaySound((LPCSTR) "C:\\cygwin\\home\\visionlab\\workspace\\cncsvision\\data\\beep\\spoken-one.wav", NULL, SND_FILENAME | SND_ASYNC);
		break;
	case 4:
		PlaySound((LPCSTR) "C:\\cygwin\\home\\visionlab\\workspace\\cncsvision\\data\\beep\\spoken-two.wav", NULL, SND_FILENAME | SND_ASYNC);
		break;


	}
	return;
}
void idle()
{
	online_trial();
	online_apparatus_alignment();
	ElapsedTime = trial_time.getElapsedTimeInMilliSec();

}


/*** Online operations ***/
void online_apparatus_alignment()
{
	updateTheMarkers();
	// mirror alignment check
	mirrorAlignment = asin(
		abs((markers.at(mirror1).p.z() - markers.at(mirror2).p.z())) /
		sqrt(
			pow(markers.at(mirror1).p.x() - markers.at(mirror2).p.x(), 2) +
			pow(markers.at(mirror1).p.z() - markers.at(mirror2).p.z(), 2)
		)
	) * 180 / M_PI;

	// screen Y alignment check
	screenAlignmentY = asin(
		abs((markers.at(screen1).p.y() - markers.at(screen3).p.y())) /
		sqrt(
			pow(markers.at(screen1).p.x() - markers.at(screen3).p.x(), 2) +
			pow(markers.at(screen1).p.y() - markers.at(screen3).p.y(), 2)
		)
	) * 180 / M_PI;

	// screen Z alignment check
	screenAlignmentZ = asin(
		abs(markers.at(screen1).p.z() - markers.at(screen2).p.z()) /
		sqrt(
			pow(markers.at(screen1).p.x() - markers.at(screen2).p.x(), 2) +
			pow(markers.at(screen1).p.z() - markers.at(screen2).p.z(), 2)
		)
	) * 180 / M_PI *
		abs(markers.at(screen1).p.x() - markers.at(screen2).p.x()) /
		(markers.at(screen1).p.x() - markers.at(screen2).p.x());

}



void drawFixation(double dispDistJitter) {
	// draws a small fixation cross at the center of the display
	glDisable(GL_TEXTURE_2D);
	glColor3f(0.7f, 0.0f, 0.0f);
	glLineWidth(2.f);
	glLoadIdentity();
	glTranslated(0, 0, display_distance + dispDistJitter);

	double cross_x = 4;
	double cross_y = 4;
	glBegin(GL_LINES);
	glVertex3d(cross_x / 2. , 0, 0);
	glVertex3d(-cross_x / 2 , 0, 0);
	glVertex3d(0, -cross_y / 2., 0);
	glVertex3d(0, cross_y / 2., 0);
	glEnd();

}

// the edge contour (how curve the edge is) is an extra depth cue, we draw panels to hide them
void drawBlockingPanels(double dispDistJitter) {

	double panel_w = 50, panel_h = 65;
	double panel_separation = tan((DEG2RAD * visual_angle) / 2) * 2 * (abs(display_distance + dispDistJitter + 7));

	glLoadIdentity();

	glTranslated(0, 0, display_distance + dispDistJitter + 7);
	if (cylHorizontal == 0) {
		glRotatef(90.0, 0.0, 0.0, 1.0); // 0, 0, 1
	}

	if (panelState == 2) {
		glColor3f(0.3, 0.0f, 0.0f);
	}
	else {
		glColor3f(0.0f, 0.0f, 0.0f);
	}

	// left panel
	glBegin(GL_QUADS);
	glVertex3f(-panel_separation / 2 - panel_w,   panel_h,  0.0f);
	glVertex3f(-panel_separation / 2,             panel_h,  0.0f);
	glVertex3f(-panel_separation / 2,            -panel_h,  0.0f);
	glVertex3f(-panel_separation / 2 - panel_w,  -panel_h,  0.0f);
	glEnd();

	// right panel
	glBegin(GL_QUADS);
	glVertex3f(panel_separation / 2,              panel_h,  0.0f);
	glVertex3f(panel_separation / 2 + panel_w,    panel_h,  0.0f);
	glVertex3f(panel_separation / 2 + panel_w,   -panel_h,  0.0f);
	glVertex3f(panel_separation / 2,             -panel_h,  0.0f);
	glEnd();

	/*
	// top panel
	glBegin(GL_QUADS);
	glVertex3f(-panel_separation / 2,             panel_h,   0.0f);
	glVertex3f( panel_separation / 2,             panel_h,   0.0f);
	glVertex3f( panel_separation / 2, panel_separation / 2,  0.0f);
	glVertex3f(-panel_separation / 2, panel_separation / 2,  0.0f);
	glEnd();

	// bottom panel
	glBegin(GL_QUADS);
	glVertex3f(-panel_separation / 2, -panel_separation / 2,   0.0f);
	glVertex3f( panel_separation / 2, -panel_separation / 2,   0.0f);
	glVertex3f( panel_separation / 2,              -panel_h,   0.0f);
	glVertex3f(-panel_separation / 2,              -panel_h,   0.0f);
	glEnd();
*/

}



std::vector<Vector3d> buildStereoCylinder(double randomDotsDepth, double dispDistJitter, bool isDistractor)
{
	std::vector<Vector3d> dot_container;

	if(isDistractor){
		dot_per_col = 14, dot_per_row = 14;
		stimulus_height = tan((DEG2RAD * visual_angle) / 2) * 2 * (abs(display_distance + randomDotsDepth / 2.));			
	}else{
		dot_per_col = 16, dot_per_row = 16;
		stimulus_height = tan((DEG2RAD * visual_angle) / 2) * 2 * (abs(display_distance + dispDistJitter - randomDotsDepth));
	}

	double temp_height = 1.5 * stimulus_height, temp_width = temp_height;
	double step_x = temp_width / (dot_per_row - 1), step_y = (temp_height) / (dot_per_col - 1); //step around 3.7

	double theta = rand() % 51 + 20.0;
	double cos_theta = cos(DEG2RAD * theta), sin_theta = sin(DEG2RAD * theta);

	if (stimulus_height / 2. < randomDotsDepth) {

		double x_temp, y_temp, x, y, z;

		for (int i_y = 0; i_y < dot_per_col; i_y ++){
				
			for (int i_x = 0; i_x < dot_per_row; i_x ++){

				x_temp = i_x * step_x - temp_width / 2 + 
					(rand() % 17)/16.0 * jitter_x_max - jitter_x_max / 2;
			
				y_temp = i_y * step_y - temp_height / 2 + 
					(rand() % 17)/16.0 * jitter_y_max - jitter_y_max / 2;

				x = x_temp * cos_theta + y_temp * sin_theta;
				y = -x_temp * sin_theta + y_temp * cos_theta;
				
				if(abs(y) < stimulus_height / 2){

					z = randomDotsDepth * sqrt(1 - pow(y / (stimulus_height / 2.), 2));

					dot_container.push_back(Vector3d(x, y, z));
				
				}
			}
		}
	}
	else {
		double R = (pow(randomDotsDepth, 2) + pow((stimulus_height / 2.), 2)) / (2 * randomDotsDepth);

		double x_temp, y_temp, x, y, z;

		for (int i_y = 0; i_y < dot_per_col; i_y ++){
				
			for (int i_x = 0; i_x < dot_per_row; i_x ++){

				x_temp = i_x * step_x - temp_width / 2 + 
					(rand() % 17)/16.0 * jitter_x_max - jitter_x_max / 2;
			
				y_temp = i_y * step_y - temp_height / 2 + 
					(rand() % 17)/16.0 * jitter_y_max - jitter_y_max / 2;

				x = x_temp * cos_theta + y_temp * sin_theta;
				y = -x_temp * sin_theta + y_temp * cos_theta;

					if(abs(y) < stimulus_height / 2 ){

						z =  sqrt(pow(R, 2) - pow(y, 2)) - R + randomDotsDepth;

						dot_container.push_back(Vector3d(x, y, z));
				
					}

			}
		}
		
	}
	return dot_container;
}



std::vector<Vector3d> buildStereoGaussian(double randomDotsDepth, double dispDistJitter, bool isDistractor)
{
	std::vector<Vector3d> dot_container;

	if(isDistractor){
		dot_per_col = 14, dot_per_row = 14;
		stimulus_height = tan((DEG2RAD * visual_angle) / 2) * 2 * (abs(display_distance + randomDotsDepth / 2.));			
	}else{
		dot_per_col = 16, dot_per_row = 16;
		stimulus_height = tan((DEG2RAD * visual_angle) / 2) * 2 * (abs(display_distance + dispDistJitter - randomDotsDepth));
	}

	Gaussian_sig = Gaussian_sig_height_ratio * stimulus_height;

	double temp_height = 1.5 * stimulus_height, temp_width = temp_height;
	double step_x = temp_width / (dot_per_row - 1), step_y = (temp_height) / (dot_per_col - 1); //step around 3.7

	double theta = rand() % 51 + 20.0;
	double cos_theta = cos(DEG2RAD * theta), sin_theta = sin(DEG2RAD * theta);


	double x_temp, y_temp, x, y, z;

	for (int i_y = 0; i_y < dot_per_col; i_y ++){
			
		for (int i_x = 0; i_x < dot_per_row; i_x ++){

			x_temp = i_x * step_x - temp_width / 2 + 
				(rand() % 17)/16.0 * jitter_x_max - jitter_x_max / 2;
		
			y_temp = i_y * step_y - temp_height / 2 + 
				(rand() % 17)/16.0 * jitter_y_max - jitter_y_max / 2;

			x = x_temp * cos_theta + y_temp * sin_theta;
			y = -x_temp * sin_theta + y_temp * cos_theta;
			
			if(abs(y) < stimulus_height / 2){

				z = randomDotsDepth * exp(-((pow(y,2))/(2*pow(Gaussian_sig,2))));

				dot_container.push_back(Vector3d(x, y, z));
			
			}
		}
	}
	
	return dot_container;
}



std::vector<Vector3d> buildStereoCosine(double randomDotsDepth, double dispDistJitter, bool isDistractor){
		
	std::vector<Vector3d> dot_container;

	if(isDistractor){
		dot_per_col = 14, dot_per_row = 14;
		stimulus_height = tan((DEG2RAD * visual_angle) / 2) * 2 * (abs(display_distance + randomDotsDepth / 2.));			
	}else{
		dot_per_col = 16, dot_per_row = 16;
		stimulus_height = tan((DEG2RAD * visual_angle) / 2) * 2 * (abs(display_distance + dispDistJitter - randomDotsDepth));
	}

	double temp_height = 1.5 * stimulus_height, temp_width = temp_height;
	double step_x = temp_width / (dot_per_row - 1), step_y = (temp_height) / (dot_per_col - 1); //step around 3.7

	double theta = rand() % 51 + 20.0;
	double cos_theta = cos(DEG2RAD * theta), sin_theta = sin(DEG2RAD * theta);


	double x_temp, y_temp, x, y, z, phase_y;

	for (int i_y = 0; i_y < dot_per_col; i_y ++){
			
		for (int i_x = 0; i_x < dot_per_row; i_x ++){

			x_temp = i_x * step_x - temp_width / 2 + 
				(rand() % 17)/16.0 * jitter_x_max - jitter_x_max / 2;
		
			y_temp = i_y * step_y - temp_height / 2 + 
				(rand() % 17)/16.0 * jitter_y_max - jitter_y_max / 2;

			x = x_temp * cos_theta + y_temp * sin_theta;
			y = -x_temp * sin_theta + y_temp * cos_theta;
			
			if(abs(y) < stimulus_height / 2){

				phase_y = M_PI * y / (stimulus_height / 2.0);
				z = (randomDotsDepth /2) * cos(phase_y) + (randomDotsDepth /2);

				dot_container.push_back(Vector3d(x, y, z));
			
			}
		}
	}

	return dot_container;

}

void inputRandomDots_forBlur(double randomDotsDepth, double dispDistJitter, std::vector<Vector3d> dot_container, bool isDistractor)
{
	if (dots_built) {
		glLoadIdentity();
		glColor3f(1.0f, 0.0f, 0.0f);

		if (isDistractor) {
			glTranslated(0, 0, display_distance + randomDotsDepth / 2.);
			if (cylHorizontal == 0) {
				glRotatef(90.0, 0.0, 0.0, 1.0); // 0, 0, 1
			}


			for (int i = 0; i < int(dot_container.size()); i++)
			{
				Vector3d dot_vector = dot_container.at(i);
				double x_axis = dot_vector[0];
				double y_axis = dot_vector[1];
				double z_axis = -dot_vector[2];

				glPushMatrix();
				glTranslated(x_axis, y_axis, z_axis);

				double dot_size = tan(dot_radius_blur) * abs(display_distance + randomDotsDepth + z_axis);
				glutSolidSphere(dot_size, 10, 10);
				glPopMatrix();
			}

			if (panelState > 0) {
				drawBlockingPanels(randomDotsDepth);
			}

		}
		else {
			glTranslated(0, 0, display_distance + dispDistJitter - randomDotsDepth);
			if (cylHorizontal == 0) {
				glRotatef(90.0, 0.0, 0.0, 1.0); // 0, 0, 1
			}

			//glColor3f(1.0f, 0.0f, 0.0f);
			for (int i = 0; i < int(dot_container.size()); i++)
			{
				Vector3d dot_vector = dot_container.at(i);
				double x_axis = dot_vector[0];
				double y_axis = dot_vector[1];
				double z_axis = dot_vector[2];

				glPushMatrix();
				glTranslated(x_axis, y_axis, z_axis);

				double dot_size = dot_size = tan(dot_radius_blur) * abs(display_distance - randomDotsDepth + dispDistJitter + z_axis);
				glutSolidSphere(dot_size, 10, 10);
				glPopMatrix();
			}

			if (panelState > 0) {
				drawBlockingPanels(dispDistJitter);
			}
		}

	}

}


void drawRandomDots(float dotBrightness, double randomDotsDepth, double dispDistJitter, std::vector<Vector3d> dot_container, bool isDistractor )
{
	if (dots_built) {
		glLoadIdentity();
		glColor3f(dotBrightness, 0.0f, 0.0f);

		if (isDistractor) {
			glTranslated(0, 0, display_distance + randomDotsDepth / 2.);
			if (cylHorizontal == 0) {
				glRotatef(90.0, 0.0, 0.0, 1.0); // 0, 0, 1
			}

			//glColor3f(dotBrightness, 0.0f, 0.0f);
			for (int i = 0; i < int(dot_container.size()); i++)
			{
				Vector3d dot_vector = dot_container.at(i);
				double x_axis = dot_vector[0];
				double y_axis = dot_vector[1];
				double z_axis = -dot_vector[2];

				glPushMatrix();
				glTranslated(x_axis, y_axis, z_axis);

				double dot_size = tan(DEG2RAD * dot_visangle / 2) * abs(display_distance + randomDotsDepth + z_axis);
				glutSolidSphere(dot_size, 10, 10);
				glPopMatrix();
			}

			if (panelState > 0) {
				drawBlockingPanels(randomDotsDepth);
			}

		}
		else {
			glTranslated(0, 0, display_distance + dispDistJitter - randomDotsDepth);
			if (cylHorizontal == 0) {
				glRotatef(90.0, 0.0, 0.0, 1.0); // 0, 0, 1
			}

			//glColor3f(dotBrightness, 0.0f, 0.0f);
			for (int i = 0; i < int(dot_container.size()); i++)
			{
				Vector3d dot_vector = dot_container.at(i);
				double x_axis = dot_vector[0];
				double y_axis = dot_vector[1];
				double z_axis = dot_vector[2];

				glPushMatrix();
				glTranslated(x_axis, y_axis, z_axis);

				double dot_size = tan(DEG2RAD * dot_visangle / 2) * abs(display_distance - randomDotsDepth + dispDistJitter + z_axis);
				glutSolidSphere(dot_size, 10, 10);
				glPopMatrix();
			}

			if (panelState > 0) {
				drawBlockingPanels(dispDistJitter);
			}
		}

	}

}



void drawStimulus_clear_eye()
{
	
	switch (currentStage) {

	case constantPresent:

		if (brightnessLevel_first < 1) { //dim and blurred
			drawRandomDots(0.6, depth_constPres, 0, dotContainer_first, is_distractor);
		}
		else {
			drawRandomDots(brightness_constPres, depth_constPres, 0, dotContainer_first, is_distractor);
		}	
		break;


	case firstFixation:
		drawFixation(distDist_jitter_first);
		break;

	case firstPresent:

		drawRandomDots(brightness_first, depth_first, distDist_jitter_first, dotContainer_first, false);

		break;

	case interStimBreak:
		drawFixation(distDist_jitter_second);
		break;

	case secondPresent:

		drawRandomDots(brightness_second, depth_second, distDist_jitter_second, dotContainer_second, false);

		break;

	case distractorPresent:

		if (ElapsedTime > blankTime_beforeDistractor ) { //because ElapsedTime get reset when repond 1 or 2

			drawRandomDots((brightness_first + brightness_second)/2, depth_distractor, 0, dotContainer_distractor, true);
	
		}

		break;


	case breakTime:
		drawProgressBar();
		break;

	}

}



void drawStimulus_blur_eye()
{

	switch (currentStage) {

	case constantPresent:

		if (brightnessLevel_first < 1) {
			drawRandomDots_blurred(depth_constPres, 0, dotContainer_first, is_distractor); // for blur eye when blurred and dimmed
		}
		else {
			drawRandomDots(brightness_constPres, depth_constPres, 0, dotContainer_first, is_distractor);
		}
		break;


	case firstFixation:
		drawFixation(distDist_jitter_first);
		break;

	case firstPresent:

		if (brightnessLevel_first < 1) {
			drawRandomDots_blurred(depth_first, distDist_jitter_first, dotContainer_first, false); // for blur eye when blurred and dimmed
		}
		else {
			drawRandomDots(brightness_first, depth_first, distDist_jitter_first, dotContainer_first, false);
		}
		break;

	case interStimBreak:
		drawFixation(distDist_jitter_second);
		break;

	case secondPresent:

		if (brightnessLevel_second < 1) {
			drawRandomDots_blurred(depth_second, distDist_jitter_second, dotContainer_second, false); // for blur eye when blurred and dimmed
		}
		else {
			drawRandomDots(brightness_second, depth_second, distDist_jitter_second, dotContainer_second, false);
		}


		break;

	case distractorPresent:

		if (ElapsedTime > blankTime_beforeDistractor) { //because ElapsedTime get reset when repond 1 or 2

			// distractors are always clear
			drawRandomDots((brightness_first + brightness_second) / 2, depth_distractor, 0, dotContainer_distractor, true);

		}

		break;



	case breakTime:
		drawProgressBar();
		break;


	}

}



void initTrial()
{
	dots_built = false;

	if (!dots_built) {

		if (currentStage == constantPresent) {

			dot_radius_blur = atan(dot_size_bfBlur / abs(display_distance - 30));

			switch(expShape){
				case cylinder:
					dotContainer_first = buildStereoCylinder(depth_constPres, 0, is_distractor);
				break;

				case gaussian:
					dotContainer_first = buildStereoGaussian(depth_constPres, 0, is_distractor);
				break;

				case cosine:
					dotContainer_first = buildStereoCosine(depth_constPres, 0, is_distractor);
				break;
			}
			
			dots_built = true;

		}
		else {

			currentStage = preparation;

			distDist_jitter_first = rand() % 7 - 3;
			distDist_jitter_second = rand() % 7 - 3;


			if (training) {

				brightnessLevel_first = rand() % 2;
				brightnessLevel_second = rand() % 2;

				depth_first = rand() % 20 + 20.0;
				depth_second = rand() % 20 + 20.0;

			}
			else {
				stairID = trial.getCurrent().second->getCurrentStaircase()->getID();
				stair_reversal = trial.getCurrent().second->getCurrentStaircase()->getReversals();
				ascending = trial.getCurrent().second->getCurrentStaircase()->getAscending();

				brightnessLevel_comp = trial.getCurrent().first["compBrightLvl"];
				brightness_std = 0.6 + brightnessLevel_std * 0.4;
				brightness_comp = 0.6 + brightnessLevel_comp * 0.4;

				depthLevel_std = trial.getCurrent().first["stdDepthLvl"];

				if (sessionNum == 1) {
					depth_std = 20.0 + depthLevel_std * 8.0;
				}
				else {					
						if (depthLevel_std == 0) {
							depth_std = flatStdDepth_dimStd;
						}
						else {
							depth_std = deepStdDepth_dimStd;
						}
				}
				depth_comp = trial.getCurrent().second->getCurrentStaircase()->getState();


				if (rand() % 2 == 0) {
					stdFirst = true;
					brightnessLevel_first = brightnessLevel_std;
					brightnessLevel_second = brightnessLevel_comp;
					depth_first = depth_std;
					depth_second = depth_comp;

				}
				else {
					stdFirst = false;
					brightnessLevel_first = brightnessLevel_comp;
					brightnessLevel_second = brightnessLevel_std;
					depth_first = depth_comp;
					depth_second = depth_std;
				}

			}

			brightness_first = 0.6 + brightnessLevel_first * 0.4;
			brightness_second = 0.6 + brightnessLevel_second * 0.4;
			
			dotContainer_first = buildStereoCylinder(depth_first, distDist_jitter_first, false);
			dotContainer_second = buildStereoCylinder(depth_second, distDist_jitter_second, false);
			//dotContainer_first = buildStereoCylinderOld(depth_first);
			//dotContainer_second = buildStereoCylinderOld(depth_second);

			depth_distractor = (depth_first + depth_second)/2;
			dotContainer_distractor = buildStereoCylinder(depth_distractor, 0, true);
			dots_built = true;

		}
	}


}


void online_trial() {

	switch (currentStage) {

	case preparation:

		if (dots_built) {
			// reset TrialTime variables
			ElapsedTime = 0;
			presentTime = 0;
			respondTime = 0;
			trial_time.reset();
			trial_time.start(); // start the timer
			currentStage = firstFixation;
			
		}
		break;

	case firstFixation:

		if (ElapsedTime >  fixationTime) {
			beepOk(3);
			currentStage = firstPresent;
		}
		break;

	case firstPresent:

		if (ElapsedTime > fixationTime + presentationTime) {
			
			currentStage = interStimBreak;

		}
		break;

	case interStimBreak:

		if (ElapsedTime > fixationTime + presentationTime + fixationTime) {
			beepOk(4);
			currentStage = secondPresent;
		}
		break;

	case secondPresent:

		if (ElapsedTime > fixationTime + 2 * presentationTime + fixationTime) {
			presentTime = ElapsedTime;
			currentStage = respond;
		}
		break;


	case distractorPresent:
	
		if (ElapsedTime > blankTime_beforeDistractor + distractorTime ) { //because ElapsedTime get reset when repond 1 or 2
			
			currentStage = waitForNextTrial;
		}
		break;

	case waitForNextTrial:

		if (ElapsedTime > blankTime_beforeDistractor + distractorTime + blankTime_afterDistractor) { //because ElapsedTime get reset when repond 1 or 2
			advanceTrial();
		}
		break;

	}

}



void advanceTrial()
{
	// "subjName\tIOD\tblockN\ttrialN\tdisplayDistance\tvisualAngle\tdotNum\tisRoated\tstdBrightness\tcompBrightness\tstdDepth\tcompDepth\tresp_firstDeeper\tresp_compDeeper\tID\treversals\tascending\tRT";
	if (training) {

		initTrial();
	}
	else {
		responseFile << fixed <<
			subjectName << "\t" <<		//subjName
			interoculardistance << "\t" <<
			blkNum << "\t" <<
			trialNum << "\t" <<
			display_distance << "\t" <<
			visual_angle << "\t" <<
			dot_number << "\t" <<
			cylHorizontal << "\t" <<
			brightness_std << "\t" <<
			brightness_comp << "\t" <<
			depth_std << "\t" <<
			depth_comp << "\t" <<
			resp_firstDeeper << "\t" <<
			resp_compDeeper << "\t" <<
			stairID << "\t" <<
			stair_reversal << "\t" <<
			ascending << "\t" <<
			respondTime << endl;

			trial.next(resp_compDeeper);

			if(trial.isEmpty()){

				beepOk(1);
				visibleInfo = true;
				currentStage = expFinished;

			}
			else {
				if(trialNum % 60 == 0){
					trialNum++;
					percentComplete = trialNum /2.4;
					visibleInfo = true;
					currentStage = breakTime;

				}
				else{
					trialNum++;
					initTrial();
				}
		}

	}

}

// Funzione che gestisce il ridimensionamento della finestra
void handleResize(int w, int h)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

}
// Funzione di callback per gestire pressioni dei tasti
void handleKeypress(unsigned char key, int x, int y)
{
	//cout << "listening for keypress" << endl;
	switch (key)
	{

	case 'Q':
	case 'q':
	case 27:	//corrisponde al tasto ESC
	{

		shutdown();
	}
	break;


	case 'i':
		visibleInfo = !visibleInfo;
		break;


	case 't': // change texture
	{
		if (training) {
			training = false;
			beepOk(1);
			trialNum = 1;
			visibleInfo = true;
			currentStage = breakTime;
			
		}
	}
	break;

	case '7':
		ElapsedTime = 0;
		trial_time.reset();
		trial_time.start();
		currentStage = firstFixation;
		break;

	case '+':
	{
		if (currentStage == constantPresent) {
			beepOk(0);
			visibleInfo = false;
			currentStage = preparation;
			initTrial();
		}

		if (currentStage == breakTime) {
			beepOk(0);
			visibleInfo = false;
			initTrial();
		}
	}
	break;


	case '1':
		if (currentStage == respond) {

			resp_firstDeeper = true;
			beepOk(0);
			if (stdFirst)
				resp_compDeeper = false;
			else
				resp_compDeeper = true;

			respondTime = ElapsedTime - presentTime; 
			ElapsedTime = 0;
			trial_time.reset();
			trial_time.start();
			currentStage = distractorPresent;
		}
		break;

	case '2':
		if (currentStage == respond) {
			resp_firstDeeper = false;
			beepOk(0);
			//if the second presented stimulus was a test stimulus (first presented stimulus was a reference)
			if (stdFirst)
				resp_compDeeper = true;
			else
				resp_compDeeper = false;

			respondTime = ElapsedTime - presentTime; 
			ElapsedTime = 0;
			trial_time.reset();
			trial_time.start();
			currentStage = distractorPresent;
		}
		break;

	case '3':
		blur_one_eye = !blur_one_eye;
		break;


	/*

	////////////////////// other changes we can make with the keypresses //////////////////
	// change depth
	case '1':
	{
		if (depth_constPres > 5) {
			depth_constPres = depth_constPres - 2;
			initTrial();
		}
	}
	break;

	// change box_width
	case '2': // change box_width
	{
		dots_built = false;
		blurringBox_Width = blurringBox_Width + 1;

		glUseProgram(BlurH);
		glUniform1i(BlurH.UniformLocations[0], blurringBox_Width);
		glUseProgram(BlurV);
		glUniform1i(BlurV.UniformLocations[0], blurringBox_Width);
		glUseProgram(0);
		dots_built = true;
		
	}
	break;


	// change extra blurring passes
	case '3':
	if(blur_extraPasses > 0)
		blur_extraPasses = blur_extraPasses - 1;
	break;

	// change dot size
	case '4':
	if(dot_size_bfBlur > 0.1){
		dot_size_bfBlur = dot_size_bfBlur - 0.02;
		initTrial();
	}
	break;
	


	// change shape type
	case '5':
		expShape = (ShapeTypes)((expShape + 1) % 3);
		initTrial();
		break;

	// switch between test and distractor
	case '6':
	{
		is_distractor = !is_distractor;
		initTrial();
	}
	break;

	// change dot density
	case '7':
	{
		dot_per_col++;
		dot_per_row++;
		initTrial();
	}
	break;

	// change dot jitter
		case '8':
	{
		jitter_x_max = jitter_x_max + 0.1;
		jitter_y_max = jitter_y_max + 0.1;
		initTrial();
	}
	break;

	// switch between bright and dim constantPresent
	case '9':
		brightnessLevel_first = (brightnessLevel_first + 1) % 2;
		initTrial();
		break;

	*/

	}
}


// this is run at compilation because it's titled 'main'
int main(int argc, char* argv[])
{
	//functions from cncsvision packages
	mathcommon::randomizeStart();

	// initializes optotrak and velmex motors
	initOptotrak();
	initMotors();

	// initializing glut (to use OpenGL)
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STEREO);
	
	glutGameModeString("1024x768:32@85"); //resolution  
	glutEnterGameMode();
	glutFullScreen();


	// initializing experiment's parameters

	initRendering(); // initializes the openGL parameters needed for creating the stimuli

	glewInit();
	initBlur();
	

	initStreams(); // streams as in files for writing data

	initVariables();

	// glut callback, OpenGL functions that are infinite loops to constantly run 

	glutDisplayFunc(drawGLScene); // keep drawing the stimuli

	glutKeyboardFunc(handleKeypress); // check for keypress

	glutReshapeFunc(handleResize);

	glutIdleFunc(idle);

	glutTimerFunc(TIMER_MS, update, 0);

	glutSetCursor(GLUT_CURSOR_NONE);

	boost::thread initVariablesThread(&initVariables);

	glutMainLoop();
	// When the program exists, clean up
	cleanup();
	return 0;
}