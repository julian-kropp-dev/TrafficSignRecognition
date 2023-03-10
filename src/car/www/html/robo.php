<html>
<head>
  <meta name="viewport" content="width=device-width" />
  <title>Control</title>
</head>
<body>
  <form method="get" action="robo.php">
    <input type="submit" value="F" name="type">
<input type="submit" value="B" name="type">
<input type="submit" value="L" name="type">
<input type="submit" value="G" name="type">
<input type="submit" value="R" name="type">
    <input type="number" value="0" name="speed">
  </form>
  <?php
echo $_GET['speed'];
  if($_GET['type'] == "F"){
    $gpio_on = shell_exec("sudo python /var/www/html/backwheels.py " . $_GET['speed'] . " " . $_GET['type']);
    echo "Vorwärts";
  }
  else if($_GET['type'] == "B"){
    $gpio_off = shell_exec("sudo python /var/www/html/backwheels.py " . $_GET['speed'] . " " . $_GET['type']);
    echo "Rückwärts";
  }
else if($_GET['type'] == "R"){
    $gpio_off = shell_exec("sudo python /var/www/html/steering.py " . $_GET['type']);
    echo "Rechts";
  }
else if($_GET['type'] == "L"){
    $gpio_off = shell_exec("sudo python /var/www/html/steering.py " . $_GET['type']);
    echo "Links";
  }
else if($_GET['type'] == "G"){
    $gpio_off = shell_exec("sudo python /var/www/html/steering.py " . $_GET['type']);
    echo "Geradeaus";
  }

  ?>
</body>
</html>
