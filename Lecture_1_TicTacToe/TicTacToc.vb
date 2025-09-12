START

INPUT nameX  ← "Enter name for Player X: "
INPUT nameO  ← "Enter name for Player O: "

board ← array[1..9] initialized to EMPTY
currentPlayer ← 'X'
currentName ← nameX
moves ← 0

WHILE TRUE:
    PRINT currentName + " (" + currentPlayer + "), choose a position (1-9): "
    INPUT pos

    IF pos < 1 OR pos > 9 THEN
        PRINT "Invalid position, try again."
        CONTINUE

    IF board[pos] is not EMPTY THEN
        PRINT "Position already taken, choose another."
        CONTINUE

    board[pos] ← currentPlayer
    moves ← moves + 1

    IF checkWin(board, currentPlayer) THEN
        PRINT currentName + " (" + currentPlayer + ") wins!"
        BREAK

    IF moves == 9 THEN
        PRINT "It's a draw!"
        BREAK

    // Switch player
    IF currentPlayer == 'X' THEN
        currentPlayer ← 'O'
        currentName ← nameO
    ELSE
        currentPlayer ← 'X'
        currentName ← nameX
    END IF
END WHILE


FUNCTION checkWin(board, symbol):
    winningLines = [
        (1,2,3), (4,5,6), (7,8,9),
        (1,4,7), (2,5,8), (3,6,9),
        (1,5,9), (3,5,7)
    ]

    FOR each (a,b,c) in winningLines:
        IF board[a] == symbol AND board[b] == symbol AND board[c] == symbol THEN
            RETURN TRUE
    RETURN FALSE
