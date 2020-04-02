var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  el("upload-label").innerHTML = input.files[0].name;
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
    el("image-picked").className = "";
  };
  reader.readAsDataURL(input.files[0]);
}

function analyze() {
  var uploadFiles = el("file-input").files;
  if (uploadFiles.length !== 1) alert("Please select a file to analyze!");

  el("analyze-button").innerHTML = "Analyzing...";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,
    true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      var response = JSON.parse(e.target.responseText);
      
      el("results").style = "display: visible;";
      el("progressbar_1").innerHTML = `${response["results"][0][1]} %`;
      el("progressbar_2").innerHTML = `${response["results"][1][1]} %`;
      el("progressbar_3").innerHTML = `${response["results"][2][1]} %`;
      el("progressbar_4").innerHTML = `${response["results"][3][1]} %`;

      el("label_1").innerHTML = `${response["results"][0][0]}`;
      el("label_2").innerHTML = `${response["results"][1][0]}`;
      el("label_3").innerHTML = `${response["results"][2][0]}`;
      el("label_4").innerHTML = `${response["results"][3][0]}`;

      el("progressbar_1").setAttribute('style', 'width:'+Number(response["results"][0][1])+'%')
      el("progressbar_1").setAttribute('aria-valuenow', response["results"][0][1]);
      el("progressbar_2").setAttribute('style', 'width:'+Number(response["results"][1][1])+'%')
      el("progressbar_2").setAttribute('aria-valuenow', response["results"][1][1]);
      el("progressbar_3").setAttribute('style', 'width:'+Number(response["results"][2][1])+'%')
      el("progressbar_3").setAttribute('aria-valuenow', response["results"][2][1]);
      el("progressbar_4").setAttribute('style', 'width:'+Number(response["results"][3][1])+'%')
      el("progressbar_4").setAttribute('aria-valuenow', response["results"][3][1]);
    }
    el("analyze-button").innerHTML = "Analyze";
  };

  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}