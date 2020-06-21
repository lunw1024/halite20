var game;
var extra;
var turn = 0;
var me = 0

$(document).ready(function () {

    //Init

    $(function () {
        $("#slider").slider({
            min: 1,
            max: 400,
            change: update
        });
    });

    if (!window.File || !window.FileReader || !window.FileList || !window.Blob) {
        alert('The File APIs are not fully supported in this browser.');
        return;
    }

    //File Loader
    $('#start').bind('click', function (event) {
        var gameLoad = document.getElementById('game-load');
        var extraLoad = document.getElementById('extra-load');
        ready = false;
        if (!gameLoad) {
            alert("Um, couldn't find the fileinput element.");
        }
        else if (!gameLoad.files) {
            alert("This browser doesn't seem to support the `files` property of file inputs.");
        }
        else if (!gameLoad.files[0]) {
            alert("Please select a file before clicking 'Load'");
        }
        else {
            console.log("Input detected. Launching")

            var gameReader = new FileReader()
            var extraReader = new FileReader()

            //Read Game
            gameReader.readAsText(gameLoad.files[0], 'UTF-8');
            gameReader.onload = gameReaderEvent => {
                game = JSON.parse(gameReaderEvent.target.result)
                turn = 0
                gameRefresh()
            }

            //TODO: Finish extra info

            if (extraLoad.files[0]) {
                extraReader.readAsText(extraLoad.files[0], 'UTF-8');
                extraReader.onload = extraReaderEvent => {
                    extra = JSON.parse(extraReaderEvent.target.result)
                }
            }
        }
    });

    //Returns generated HTML in String form
    //Next time, use a templating language. This is painful.
    function generate(n) {
        var html =
            [
                '<div class="square" type="button" class="btn btn-primary" data-toggle="modal" data-target="#modal'+n+'">',
                '<div>',
                '<div class="modal fade" id="modal'+n+'" tabindex="-1" role="dialog" aria-labelledby="modalLabel'+n+'" aria-hidden="true">',
                    '<div class="modal-dialog" role="document">',
                        '<div class="modal-content">',
                        '<div class="modal-header">',
                            '<h5 class="modal-title" id="modalLabel'+n+'">Square '+Math.floor(n/game.configuration.size)+' '+n%game.configuration.size+'</h5>',
                            '<button type="button" class="close" data-dismiss="modal" aria-label="Close">',
                            '<span aria-hidden="true">&times;</span>',
                            '</button>',
                        '</div>',
                        '<div class="modal-body">',
                            '<i class="fas fa-gem" style="color:white;"></i>',
                            '<p id="haliteLabel'+n+'">0</p>',
                        '</div>',
                        '<div class="modal-footer">',
                            '<button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>',
                        '</div>',
                        '</div>',
                    '</div>',
                '</div>',
            ].join("\n")
        return html
    }

    function gameRefresh() {
        console.log("Game Refresh starting")
        if (!game) {
            console.log("Game not found")
            return
        }
        board = $('#board')
        turn = 0
        
        //Add squares
        $('#board').empty()
        for (i = 0; i < Math.pow(game.configuration.size, 2); i++) {
            board.append(generate(i));
        }
        //Calculate square size
        var n = " " + (100 / game.configuration.size).toString() + "%"
        var modifier = ""
        for (i = 0; i < game.configuration.size; i++) {
            modifier = modifier + n
        }
        board.css('grid-template-columns', modifier)
        board.css('grid-template-rows', modifier)

        update()

    }

    function getSquare(x, y) {
        var num = x * game.configuration.size + y
        return $('#board').children().eq(num)
    }

    function update() {
        //Clear
        $('.refresh').remove()
        var turn = $("#slider").slider("value")
        $('#turn-display').text(turn.toString())

        if (!game) {
            return
        }
        //Set turn limit
        if (turn > game.steps.length) {
            turn = game.steps.length - 1
        }
        turn = turn - 1

        var turnInfo = game.steps[turn]
        var haliteMap = turnInfo[me].observation.halite

        //Loop through halite map and adjust colors
        for (i = 0; i < haliteMap.length; i++) {
            var color = haliteMap[i] / game.configuration.maxCellHalite * 255
            $('#board').children().eq(i).css('background-color', 'rgb(' + color + ',' + color + ',' + color + ')')
            $('#haliteLabel'+i).text(haliteMap[i])
        }


        var colors = ['blue', 'red', 'green', 'yellow']
        var bgColors = ['cyan','pink','lime','khaki']

        //Loop through units and stuff
        //console.log(turnInfo[me].observation.players[0])

        for (team = 0; team < turnInfo[me].observation.players.length; team++) {
            var halite = turnInfo[me].observation.players[team][0]
            var ships = turnInfo[me].observation.players[team][2]
            var shipyards = turnInfo[me].observation.players[team][1]
            var color = colors[team]

            $('#haliteTotal'+team).text(halite)

            for (var ship in ships) {
                if (ships[ship][1] == 0) {
                    $('#board').children().eq(ships[ship][0]).append('<i class="fas fa-fighter-jet refresh" style="color:' + color + ';"></i>')
                } else {
                    $('#board').children().eq(ships[ship][0]).append('<i class="fas fa-tractor refresh" style="color:' + color + ';"></i>')
                }
            }
            for (var yard in shipyards) {
                console.log($('#board').children().eq(shipyards[yard]))
                $('#board').children().eq(shipyards[yard]).css('background-color', bgColors[team])
            }
        }
    }

});
