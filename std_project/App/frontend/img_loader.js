let ls;
$("document").ready(() => {
    ls = window.localStorage,
        photo = document.getElementById('uploadFile'),
        canvas = document.getElementById('canvasOrig'),
        canvas2 = document.getElementById('canvasNew'),
        context = canvas.getContext('2d'),
        context2 = canvas2.getContext('2d'),
        fileReader = new FileReader(),
        img = new Image();

    fileReader.onload = (e) => {
        img.src = e.target.result;
    };

    img.onload = () => {
        drawImage(img, canvas, context);
    };

    photo.addEventListener('change', function() {
        var file = this.files[0];
        return file && fileReader.readAsDataURL(file);
    });
});

function drawImage(img, canvas, context) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    var rw = img.width / canvas.width;
    var rh = img.height / canvas.height;
    var h, w;

    if (rw > rh) {
        h = Math.round(img.height / rw);
        w = canvas.width;
    } else {
        w = Math.round(img.width / rh);
        h = canvas.height;
    }

    x = (canvas.width - w) / 2,
        y = (canvas.height - h) / 2;

    if (img.width) context.drawImage(img, x, y, w, h);
}

function chooseFile() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context2.clearRect(0, 0, canvas2.width, canvas2.height);

    $("#uploadFile").click();
}

function processImage() {

    //send image to server side with flask
    document.getElementById("loader").style.display = "block";
    filename = document.getElementById('uploadFile').value;
    $.post("http://localhost:8000/process", {
        img: filename
    }, function(err, req, resp) {
        //get the new image from python side
        console.log("resp=", resp["responseText"]);
        showResult(resp["responseText"]);
    });
}

function showResult(img) {
    new_img = new Image();
    new_img.src = '../' + img;
    console.log(new_img.src);
    document.getElementById("loader").style.display = "none";
    new_img.onload = () => {
        drawImage(new_img, canvas2, context2);
    };
}