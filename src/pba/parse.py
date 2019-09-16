from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional

import dateutil.parser
from lxml import etree, html
from lxml.html import HtmlElement
from tqdm import tqdm

from pba.prediction import Prediction, Response


def parse_response(response: HtmlElement) -> Response:
    """
    Possible actions:
    - comment
    - credence
    - outcome: {"right", "wrong", "unknown", "withdrawn"}
    - change-deadline
    - change-prediction
    - made-the-prediction
    - other?
    """
    # Does the case where someone does something AND judges the prediction exist?

    # Every response has, at minimum, a user and date attached
    users = response.xpath("a[@class='user']/text()")
    if users == [] and response.text.split()[0] == 'Anonymous':
        user = 'Anonymous'
    else:
        user = users[0]
    date = response.xpath("span/@title")[0]

    # We can also have (possibly multiple) "actions" in a respones, like setting a credence
    # or changing the deadline.
    actions = []
    comments = response.xpath("span[@class='comment']")
    if len(comments) > 0:
        comment_text = etree.tostring(comments[0], pretty_print=True, encoding='unicode')[22:-10]
        actions.append(("comment", comment_text))

    credences = response.xpath("span[@class='confidence']/text()")
    if credences:
        actions.append(("credence", int(credences[0][:-1])))

    outcomes = response.xpath("span[@class='outcome']/text()")
    if outcomes:
        actions.append(("outcome", outcomes[0]))

    deadline_changes = response.xpath("span[@class='date']/@title")
    if len(deadline_changes) > 1:
        actions.append(("change-deadline", deadline_changes[0]))
        date = response.xpath("span/@title")[1]

    resp_str = str(etree.tostring(response, pretty_print=True))
    change_loc = resp_str.find('changed their prediction from &#8220;')
    if change_loc != -1:
        actions.append(('change-prediction', resp_str[change_loc+37:].split('&#8221')[0]))

    withdraw_loc = resp_str.find("</a>\\nwithdrew the prediction\\n<")
    if withdraw_loc != -1:
        actions.append(('outcome', 'withdrawn'))

    make_visible_loc = resp_str.find("</a>\\nmade the prediction")
    if make_visible_loc != -1:
        actions.append(('made-visible', None))

    return Response(user, date, dict(actions))


def parse_html(page: HtmlElement) -> Dict[str, Any]:
    # outcome: {"right", "wrong", "unknown", "withdrawn"}
    title = page.xpath('//h1')[0]
    event = title.text_content().strip()
    outcome = title.values()[0]
    if outcome == "":
        outcome = "unknown"
    user = page.xpath("//div[@id='content']/p/a[@class='user']/text()")[0]

    all_times = page.xpath("//div[@id='content']/p/span[@class='date']/@title")
    assert len(all_times) == 2, "Something wrong with the dates."
    time_created = dateutil.parser.parse(all_times[0])
    time_known = dateutil.parser.parse(all_times[1])

    responses = [parse_response(resp) for resp in page.xpath("//ul[@id='responses']/*")]

    return {"event": event, "outcome": outcome, "user": user, "time_created": time_created,
            "time_known": time_known, "responses": responses}


def parse_file(file_path: str) -> Optional[Prediction]:
    with open(file_path, "r") as f:
        page = f.read()

    assert page != "", f"Page {file_path} is empty"
    page_html = html.fromstring(page)
    page_title = page_html.xpath('//title/text()')[0]
    response_types = {'PredictionBook: How sure are you?': "home",
                      "The page you were looking for doesn't exist (404)": "404",
                      "500 Internal Server Error": "500",
                      "We're sorry, but something went wrong (500)": "500",
                      "504 Gateway Time-out": "504"}
    page_type = response_types.get(page_title, "prediction")

    if page_type == "prediction":
        pred_kwargs = parse_html(page_html)
        base_path = os.path.basename(file_path)
        hyph_loc = base_path.find("-")
        pred_kwargs['number'] = int(base_path[hyph_loc+1:].split(".")[0])
        return Prediction(**pred_kwargs)
    else:
        return None


def parse_pages(page_dir: str) -> Iterator[Prediction]:
    for page_file in tqdm([f for f in os.listdir(page_dir) if f.endswith(".html")]):
        pred = parse_file(os.path.join(page_dir, page_file))
        if pred is not None:
            yield pred


def load_json_to_predictions(json_fname: str) -> List[Prediction]:
    with open(json_fname) as f:
        return [Prediction.from_dict(pred) for pred in tqdm(json.load(f))]
