$("document").ready(() => {
    let ls = window.localStorage,
        photo = document.getElementById('uploadFile'),
        canvas = document.getElementById('canvasOrig'),
        context = canvas.getContext('2d'),
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
    $("#uploadFile").click();
}

function processImage() {
    //send image to server side with flask
    $.post("http://localhost:5000/postmethod", {
        img: "hello"
    }, function(err, req, resp) {
        //get the new image from python side
        console.log("resp=", resp["responseJSON"]);
    });
}