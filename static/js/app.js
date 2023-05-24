URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording

var AudioContext = window.AudioContext || window.webkitAudioContext;

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var pauseButton = document.getElementById("pauseButton");
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
pauseButton.addEventListener("click", pauseRecording);

function startRecording() {
	console.log("recordButton clicked");
    var constraints = { audio: true, video:false }
	recordButton.disabled = true;
	stopButton.disabled = false;
	pauseButton.disabled = false;

	navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
		console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

		
		audioContext = new AudioContext();
		document.getElementById("formats").innerHTML="Format: 1 channel pcm @ "+audioContext.sampleRate/1000+"kHz"
		gumStream = stream;
		input = audioContext.createMediaStreamSource(stream);
		rec = new Recorder(input,{numChannels:1})
		rec.record()
		console.log("Recording started");

	}).catch(function(err) {
    	recordButton.disabled = false;
    	stopButton.disabled = true;
    	pauseButton.disabled = true
	});
}

function pauseRecording() {
	console.log("pauseButton clicked rec.recording=",rec.recording );
	if (rec.recording){
		rec.stop();
		pauseButton.innerHTML="Resume";
	}else{
		rec.record()
		pauseButton.innerHTML="Pause";

	}
}

function stopRecording() {
	console.log("stopButton clicked");
	stopButton.disabled = true;
	recordButton.disabled = false;
	pauseButton.disabled = true;
	pauseButton.innerHTML="Pause";
	rec.stop();
	gumStream.getAudioTracks()[0].stop();
	rec.exportWAV(createDownloadLink);
}

function createDownloadLink(blob) {

	var formData = new FormData();
	formData.append('data',blob);
	var sendAction = document.getElementById("action").value;
	
	formData.append('action',sendAction);
	
	if(sendAction == 'login'){
	
		formData.append('username',document.getElementById('username').value);	
		formData.append('password',document.getElementById("password").value);
	} 
	else {

		formData.append('name',document.getElementById('name').value);
		formData.append('username',document.getElementById('email').value);
		formData.append('mobile',document.getElementById('mobile').value);
		formData.append('college',document.getElementById('college').value);
		formData.append('dept',document.getElementById('dept').value);
		formData.append('about',document.getElementById('about').value);
		formData.append('image',document.getElementById('image').files[0]);
		formData.append('password',document.getElementById('password').value);

	}

	console.log('blob : ',blob);
	$.ajax({
        type: 'POST',
        url: '/result',
        data: formData,
        contentType: false,
        processData: false,
        success: function(result) {
          console.log('success', result);
        },
        error: function(result) {
          alert('sorry an error occured');
        }
      });
    

	var url = URL.createObjectURL(blob);
	var au = document.createElement('audio');
	var li = document.createElement('li');
	var link = document.createElement('a');
	au.controls = true;
	au.src = url;
	// link.href = url;
	// link.download = "recording.flac"; 
	// link.innerHTML = "Save to disk";
	// link.click()


}